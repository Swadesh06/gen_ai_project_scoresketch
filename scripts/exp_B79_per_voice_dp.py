"""B79 — Per-voice independent DP using B76 voice tracker (Phase D).

B78 showed that B76's 93% voice-tracker accuracy doesn't move snap-F1 in
the existing pipeline because the DP is run on ALL notes together (voice
info is only used to adjust per-note durations).

B79 fixes this by:
1. Use B76 to assign each note to a voice
2. Run viterbi_quantize_rhythm INDEPENDENTLY on each voice's notes
3. Merge per-voice quantizations back to the global note index order
4. Score with B63's actual snap metric (match_notes + duration ratio)

Pass criterion: Liszt snap > 0.10 (B63 baseline 0.053 with greedy + shared DP).
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
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import (
    VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations,
)

ASAP = Path("~/datasets/asap").expanduser()
CACHE_YMT = Path("/workspace/.cache/asap_yourmt3plus")
CHECKPOINT = Path("checkpoints/voice_transformer_b76/best.pt")
OUT_JSON = Path("reports/_exp_B79_per_voice_dp.json")
TPB = 24

PIECES = [
    "Liszt/Sonata", "Schumann/Toccata", "Chopin/Berceuse_op_57",
    "Beethoven/Piano_Sonatas/21-1",
]


class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 30000):
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
    def __init__(self, d_model: int = 192, n_heads: int = 6, n_layers: int = 6,
                 ff_dim: int = 384):
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


def load_score_beats(annotation_path: Path) -> np.ndarray:
    rows = annotation_path.read_text().strip().splitlines()
    beats = []
    for r in rows:
        parts = r.split()
        if not parts: continue
        try: beats.append(float(parts[0]))
        except ValueError: continue
    return np.array(beats)


def load_score_notes(piece: str):
    mid = pretty_midi.PrettyMIDI(str(ASAP / piece / "midi_score.mid"))
    notes = []
    for inst in mid.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort(key=lambda x: x[0])
    intervals = np.array([[n[0], n[1]] for n in notes])
    pitches = np.array([440.0 * 2 ** ((n[2] - 69) / 12) for n in notes])
    return intervals, pitches


def load_b63_cache_notes(piece: str):
    safe = piece.replace("/", "__")
    cache = CACHE_YMT / f"{safe}.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"no B63 cache for {piece}")
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


def normalise_for_tracker(arr):
    out = arr.copy()
    out[:, 0] = (out[:, 0] - 60) / 12
    if len(out) > 0:
        out[:, 1] = out[:, 1] - out[0, 1]
        out[:, 3] = out[:, 3] - out[0, 3]
    return out


def predict_voices_chunked(model, notes, chunk_size: int = 512):
    arr = np.array([[n.midi(), n.onset_s, n.offset_s - n.onset_s, n.onset_s]
                     for n in notes], dtype=np.float32)
    voices = np.zeros(len(notes), dtype=np.int64)
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i+chunk_size]
        if len(chunk) == 0: continue
        x_norm = normalise_for_tracker(chunk)
        x = torch.from_numpy(x_norm).unsqueeze(0).to("cuda")
        with torch.no_grad():
            pred = model(x).argmax(-1).squeeze(0).cpu().numpy()
        voices[i:i+chunk_size] = pred
    return voices


def shared_dp(notes, beats):
    """Current production: voice tracker for offset adjust + single DP on all notes."""
    cfg = VoiceTrackConfig(pitch_jump=adaptive_pitch_jump(notes), time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=TPB)


def per_voice_dp(notes, beats, voice_ids):
    """B79 — independent DP per voice, then merge to original order.

    voice_ids: per-note voice assignment (np.array of ints).
    Returns per-note (q_on_tatum, q_off_tatum) in the original note order.
    """
    n = len(notes)
    q_on_global = np.zeros(n, dtype=np.int64)
    q_off_global = np.zeros(n, dtype=np.int64)
    for v in set(voice_ids.tolist()):
        idx = np.where(voice_ids == v)[0]
        if len(idx) == 0: continue
        v_notes = [notes[i] for i in idx]
        v_on = np.array([n.onset_s for n in v_notes])
        v_off = np.array([n.offset_s for n in v_notes])
        # Adjust offsets per voice (cap at gap-to-next)
        v_off_adj = v_off.copy()
        order = np.argsort(v_on)
        for k in range(len(order) - 1):
            i, j = order[k], order[k + 1]
            gap = v_on[j] - v_on[i]
            if 0.05 <= gap < v_off_adj[i] - v_on[i]:
                v_off_adj[i] = v_on[i] + gap
        q_on_v, q_off_v = viterbi_quantize_rhythm(v_on, v_off_adj, beats, tatums_per_beat=TPB)
        for ki, gi in enumerate(idx):
            q_on_global[gi] = q_on_v[ki]
            q_off_global[gi] = q_off_v[ki]
    return q_on_global, q_off_global


def score_snap(notes, q_on, q_off, beats, gt_iv, gt_p):
    """B63's snap metric: fraction of matched notes whose duration is correct."""
    if not notes:
        return {"snap": 0.0, "n_matched": 0}
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([n.pitch_hz if n.pitch_hz else
                       (440.0 * 2 ** ((n.midi() - 69) / 12)) for n in notes])
    matched = mir_eval.transcription.match_notes(
        gt_iv, gt_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    if not matched:
        return {"snap": 0.0, "n_matched": 0}
    # Beat-relative durations
    avg_beat = float(np.mean(np.diff(beats)))
    pred_durs = (q_off - q_on) / TPB  # in beats
    gt_durs = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    matched_pred = np.array([pred_durs[m[1]] for m in matched])
    matched_gt = np.array([gt_durs[m[0]] for m in matched])
    # snap = fraction within ±10% of correct duration
    rel_err = np.abs(matched_pred - matched_gt) / (np.abs(matched_gt) + 1e-9)
    snap = float((rel_err < 0.10).mean())
    return {"snap": snap, "n_matched": len(matched)}


def main():
    cfg_w = {"phase": "D", "tracker_ckpt": str(CHECKPOINT)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B79_per_voice_dp",
                     config=cfg_w, tags=["B79", "asap", "per-voice-dp",
                                          "phase-d", "integration"],
                     dir="logs/wandb")
    if not CHECKPOINT.exists():
        print(f"FATAL: {CHECKPOINT} not found"); run.finish(); return
    print(f"loading B76 voice tracker from {CHECKPOINT}")
    ckpt = torch.load(str(CHECKPOINT), map_location="cuda", weights_only=False)
    model = VoiceTransformer(d_model=ckpt["config"]["d_model"],
                              n_layers=ckpt["config"]["n_layers"]).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  best train acc: {ckpt['best_acc']:.4f}")

    rows = []
    for piece in PIECES:
        print(f"\n=== {piece} ===")
        try:
            notes = load_b63_cache_notes(piece)
        except FileNotFoundError as e:
            print(f"  skip: {e}"); continue
        if not notes:
            continue
        print(f"  {len(notes)} notes")
        ann = ASAP / piece / "midi_score_annotations.txt"
        if not ann.exists():
            print(f"  no annotations"); continue
        beats = load_score_beats(ann)
        gt_iv, gt_p = load_score_notes(piece)

        # Variant A: shared DP, greedy voice tracker (current production)
        q_on_a, q_off_a = shared_dp(notes, beats)
        snap_a = score_snap(notes, q_on_a, q_off_a, beats, gt_iv, gt_p)

        # Variant B: per-voice DP, greedy voice tracker
        cfg = VoiceTrackConfig(pitch_jump=adaptive_pitch_jump(notes), time_gap_s=0.5)
        greedy_voices_lol = assign_voices(notes, cfg)
        # Convert lol to flat array
        greedy_voices = np.zeros(len(notes), dtype=np.int64)
        for v_idx, vlist in enumerate(greedy_voices_lol):
            for i in vlist:
                greedy_voices[i] = v_idx
        # Limit to 2 voices (greedy can produce many; consolidate to 2 by pitch class)
        if greedy_voices.max() > 1:
            # Consolidate: voice 0 = below median pitch, voice 1 = above
            mid_p = np.median([n.midi() for n in notes])
            greedy_voices = np.array([0 if n.midi() < mid_p else 1 for n in notes])
        q_on_b, q_off_b = per_voice_dp(notes, beats, greedy_voices)
        snap_b = score_snap(notes, q_on_b, q_off_b, beats, gt_iv, gt_p)

        # Variant C: per-voice DP, B76 voice tracker
        b76_voices = predict_voices_chunked(model, notes)
        q_on_c, q_off_c = per_voice_dp(notes, beats, b76_voices)
        snap_c = score_snap(notes, q_on_c, q_off_c, beats, gt_iv, gt_p)

        rows.append({"piece": piece,
                      "snap_shared_greedy": snap_a["snap"], "n_matched_a": snap_a["n_matched"],
                      "snap_pervoice_greedy": snap_b["snap"], "n_matched_b": snap_b["n_matched"],
                      "snap_pervoice_b76": snap_c["snap"], "n_matched_c": snap_c["n_matched"]})
        print(f"  A. shared DP / greedy:   snap={snap_a['snap']:.4f}  matched={snap_a['n_matched']}")
        print(f"  B. per-voice DP / greedy: snap={snap_b['snap']:.4f}  matched={snap_b['n_matched']}")
        print(f"  C. per-voice DP / B76:   snap={snap_c['snap']:.4f}  matched={snap_c['n_matched']}")
        print(f"  delta (C-A): {snap_c['snap'] - snap_a['snap']:+.4f}")
        wandb.log({"piece": piece, **rows[-1]})

    if rows:
        means = {k: float(np.mean([r[k] for r in rows]))
                  for k in rows[0] if k.startswith("snap_")}
        print("\nMEANS over", len(rows), "pieces:")
        for k, v in means.items():
            print(f"  {k}: {v:.4f}")
        wandb.summary.update(means)
        OUT_JSON.write_text(json.dumps({"rows": rows, "means": means, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
