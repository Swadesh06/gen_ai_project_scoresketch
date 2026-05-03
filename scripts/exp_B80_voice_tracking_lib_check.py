"""B80 — Verify the per_voice_dp library refactor matches B79's standalone result.

Calls humscribe.rhythm.voice_tracking.quantize_with_voice_tracking with the
new per_voice_dp=True + voice_assigner=B76 args. Should reproduce B79's
Chopin +1.66pp result.
"""
from __future__ import annotations
import json
import math
import pickle
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import wandb

from humscribe.notes import NoteEvent
from humscribe.rhythm.voice_tracking import quantize_with_voice_tracking

ASAP = Path("~/datasets/asap").expanduser()
CACHE_YMT = Path("/workspace/.cache/asap_yourmt3plus")
CHECKPOINT = Path("checkpoints/voice_transformer_b76/best.pt")
OUT_JSON = Path("reports/_exp_B80_lib_check.json")
TPB = 24

PIECES = [
    "Liszt/Sonata", "Schumann/Toccata", "Chopin/Berceuse_op_57",
    "Beethoven/Piano_Sonatas/21-1",
]


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=30000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VoiceTransformer(nn.Module):
    def __init__(self, d_model=192, n_heads=6, n_layers=6, ff_dim=384):
        super().__init__()
        self.feat_proj = nn.Linear(4, d_model)
        self.posenc = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                             dim_feedforward=ff_dim, dropout=0.1,
                                             batch_first=True, activation="gelu",
                                             norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        h = self.feat_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        return self.head(h)


def load_score_beats(p):
    rows = p.read_text().strip().splitlines()
    out = []
    for r in rows:
        parts = r.split()
        if not parts: continue
        try: out.append(float(parts[0]))
        except ValueError: continue
    return np.array(out)


def load_score_notes(piece):
    mid = pretty_midi.PrettyMIDI(str(ASAP / piece / "midi_score.mid"))
    notes = []
    for inst in mid.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort(key=lambda x: x[0])
    intervals = np.array([[n[0], n[1]] for n in notes])
    pitches = np.array([440.0 * 2 ** ((n[2] - 69) / 12) for n in notes])
    return intervals, pitches


def load_b63_cache(piece):
    safe = piece.replace("/", "__")
    cache = CACHE_YMT / f"{safe}.pkl"
    d = pickle.loads(cache.read_bytes())
    notes = []
    for x in d.get("notes", []):
        if isinstance(x, NoteEvent):
            notes.append(x); continue
        hz = x.get("hz")
        mid = x.get("midi")
        if hz is None and mid is not None:
            hz = 440.0 * 2 ** ((mid - 69) / 12)
        if hz is None or mid is None:
            continue
        notes.append(NoteEvent(onset_s=x["on"], offset_s=x["off"],
                                pitch_midi=mid, pitch_hz=hz,
                                velocity=x.get("vel", 80),
                                confidence=x.get("conf", 1.0)))
    return notes


def normalise(arr):
    out = arr.copy()
    out[:, 0] = (out[:, 0] - 60) / 12
    if len(out) > 0:
        out[:, 1] = out[:, 1] - out[0, 1]
        out[:, 3] = out[:, 3] - out[0, 3]
    return out


def predict_voices_chunked(model, notes, chunk_size=512):
    arr = np.array([[n.midi(), n.onset_s, n.offset_s - n.onset_s, n.onset_s]
                     for n in notes], dtype=np.float32)
    voices = np.zeros(len(notes), dtype=np.int64)
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i+chunk_size]
        if len(chunk) == 0: continue
        x = torch.from_numpy(normalise(chunk)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            pred = model(x).argmax(-1).squeeze(0).cpu().numpy()
        voices[i:i+chunk_size] = pred
    return voices


def make_b76_assigner(model):
    """Return a callable (notes) -> list[list[int]] for the B76 tracker."""
    def assigner(notes):
        voice_ids = predict_voices_chunked(model, notes)
        groups = [[], []]
        for i, vid in enumerate(voice_ids):
            groups[int(vid)].append(i)
        return [g for g in groups if g]
    return assigner


def score_snap(notes, q_on, q_off, beats, gt_iv, gt_p):
    if not notes:
        return 0.0, 0
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([n.pitch_hz if n.pitch_hz else
                       (440.0 * 2 ** ((n.midi() - 69) / 12)) for n in notes])
    matched = mir_eval.transcription.match_notes(
        gt_iv, gt_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None,
    )
    if not matched:
        return 0.0, 0
    avg_beat = float(np.mean(np.diff(beats)))
    pred_durs = (q_off - q_on) / TPB
    gt_durs = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    pd = pred_durs[[m[1] for m in matched]]
    gd = gt_durs[[m[0] for m in matched]]
    rel_err = np.abs(pd - gd) / (np.abs(gd) + 1e-9)
    return float((rel_err < 0.10).mean()), len(matched)


def main():
    cfg_w = {"phase": "D", "tracker_ckpt": str(CHECKPOINT)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B80_voice_tracking_lib_check",
                     config=cfg_w, tags=["B80", "asap", "lib-refactor", "phase-d"],
                     dir="logs/wandb")

    print(f"loading B76 from {CHECKPOINT}")
    ckpt = torch.load(str(CHECKPOINT), map_location="cuda", weights_only=False)
    model = VoiceTransformer(d_model=ckpt["config"]["d_model"],
                              n_layers=ckpt["config"]["n_layers"]).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  best train acc: {ckpt['best_acc']:.4f}")

    b76_assigner = make_b76_assigner(model)
    rows = []
    for piece in PIECES:
        print(f"\n=== {piece} ===")
        notes = load_b63_cache(piece)
        if not notes: continue
        beats = load_score_beats(ASAP / piece / "midi_score_annotations.txt")
        gt_iv, gt_p = load_score_notes(piece)
        print(f"  {len(notes)} notes")

        # A. Lib default (shared DP, greedy tracker)
        q_on_a, q_off_a = quantize_with_voice_tracking(notes, beats, tatums_per_beat=TPB)
        snap_a, n_a = score_snap(notes, q_on_a, q_off_a, beats, gt_iv, gt_p)

        # B. Lib + per_voice_dp=True, greedy tracker
        q_on_b, q_off_b = quantize_with_voice_tracking(
            notes, beats, tatums_per_beat=TPB, per_voice_dp=True)
        snap_b, n_b = score_snap(notes, q_on_b, q_off_b, beats, gt_iv, gt_p)

        # C. Lib + per_voice_dp=True + B76 voice_assigner
        q_on_c, q_off_c = quantize_with_voice_tracking(
            notes, beats, tatums_per_beat=TPB, per_voice_dp=True,
            voice_assigner=b76_assigner)
        snap_c, n_c = score_snap(notes, q_on_c, q_off_c, beats, gt_iv, gt_p)

        rows.append({"piece": piece,
                      "snap_lib_default": snap_a, "snap_lib_pvd_greedy": snap_b,
                      "snap_lib_pvd_b76": snap_c,
                      "delta_b76_minus_default": snap_c - snap_a})
        print(f"  default (shared/greedy):   {snap_a:.4f}  matched={n_a}")
        print(f"  per_voice_dp + greedy:     {snap_b:.4f}")
        print(f"  per_voice_dp + B76:        {snap_c:.4f}")
        print(f"  delta:                     {snap_c - snap_a:+.4f}")
        wandb.log({"piece": piece, "snap_default": snap_a,
                    "snap_pvd_greedy": snap_b, "snap_pvd_b76": snap_c})

    if rows:
        means = {k: float(np.mean([r[k] for r in rows]))
                  for k in rows[0] if k.startswith("snap_") or k == "delta_b76_minus_default"}
        print(f"\nMEAN over {len(rows)} pieces:")
        for k, v in means.items():
            print(f"  {k}: {v:.4f}")
        wandb.summary.update(means)
        OUT_JSON.write_text(json.dumps({"rows": rows, "means": means, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
