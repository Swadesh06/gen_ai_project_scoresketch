"""B78 — Integrate B76 voice tracker into the pipeline + measure Liszt snap.

B76 trained a Transformer voice tracker that hits 90% mean acc on held-out
Romantic ASAP (Liszt 86%, Schumann 92%, Chopin 90%, Beethoven 93%). This
script wires it into a custom voice-tracking function and re-runs the same
ASAP snap evaluation B63 used.

Comparison:
- Baseline: greedy adaptive_pj voice tracker (B49, current production)
- B78: B76 Transformer voice tracker

Pieces evaluated: same 4 Romantic held-outs from B76 — Liszt Sonata,
Schumann Toccata, Chopin Berceuse, Beethoven Piano_Sonatas/21-1.
"""
from __future__ import annotations
import json
import math
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
CACHE_BD = Path("/workspace/.cache/asap_bytedance")
CACHE_YMT = Path("/workspace/.cache/asap_yourmt3plus")
CHECKPOINT = Path("checkpoints/voice_transformer_b76/best.pt")
OUT_JSON = Path("reports/_exp_B78_voice_tracker_integration.json")


PIECES = {
    "Liszt/Sonata": "Liszt_Sonata",
    "Schumann/Toccata": "Schumann_Toccata",
    "Chopin/Berceuse_op_57": "Chopin_Berceuse_op_57",
    "Beethoven/Piano_Sonatas/21-1": "Beethoven_Piano_Sonatas_21-1",
}


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
    """First column of midi_score_annotations.txt is beat time in seconds."""
    rows = annotation_path.read_text().strip().splitlines()
    beats = []
    for r in rows:
        parts = r.split()
        if not parts: continue
        try:
            beats.append(float(parts[0]))
        except ValueError:
            continue
    return np.array(beats)


def load_score_notes(piece: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (intervals, pitches) from ASAP midi_score.mid for ground truth scoring."""
    mid = pretty_midi.PrettyMIDI(str(ASAP / piece / "midi_score.mid"))
    notes = []
    for inst in mid.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort(key=lambda x: x[0])
    intervals = np.array([[n[0], n[1]] for n in notes])
    pitches = np.array([440.0 * 2 ** ((n[2] - 69) / 12) for n in notes])
    return intervals, pitches


def normalise_for_tracker(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[:, 0] = (out[:, 0] - 60) / 12
    if len(out) > 0:
        out[:, 1] = out[:, 1] - out[0, 1]
        out[:, 3] = out[:, 3] - out[0, 3]
    return out


def predict_voices_chunked(model, notes, chunk_size: int = 512):
    """Apply Transformer voice tracker on chunks of 512 notes."""
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


def transcribe_via_cache(piece: str, backend: str = "yourmt3plus") -> list[NoteEvent]:
    """Load cached transcription from B63 (saved per piece). Falls back to recomputing.

    B63 cache stores notes as dicts: {on, off, midi, hz, vel, conf}. Convert
    to NoteEvent here so downstream voice-tracking + DP code works.
    """
    safe = piece.replace("/", "__")
    if backend == "yourmt3plus":
        cache_dir = CACHE_YMT
    else:
        cache_dir = CACHE_BD
    cache = cache_dir / f"{safe}.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"no B63 cache for {piece}; run B63 first")
    import pickle
    d = pickle.loads(cache.read_bytes())
    notes = []
    for x in d.get("notes", []):
        if isinstance(x, NoteEvent):
            notes.append(x); continue
        # Dict from B63 cache
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


def quantize_with_baseline_tracker(notes, beats):
    cfg = VoiceTrackConfig(pitch_jump=adaptive_pitch_jump(notes), time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=24)


def quantize_with_b76_tracker(notes, beats, model):
    """Use B76 Transformer to assign voices, then per-voice DP.

    `predict_voices_chunked` returns a flat array of per-note voice ids;
    `per_voice_durations` expects list[list[int]] of indices per voice.
    """
    voice_ids = predict_voices_chunked(model, notes)
    voices_grouped: list[list[int]] = [[], []]
    for i, vid in enumerate(voice_ids):
        voices_grouped[int(vid)].append(i)
    on_v, off_v = per_voice_durations(notes, voices_grouped)
    return viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=24)


def score_snap(intervals_pred, pitches_pred, intervals_ref, pitches_ref):
    if len(intervals_pred) == 0 or len(intervals_ref) == 0:
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        intervals_ref, pitches_ref, intervals_pred, pitches_pred,
        offset_ratio=None, onset_tolerance=0.05,
    )[2]


def main() -> None:
    cfg_w = {"phase": "D", "tracker_ckpt": str(CHECKPOINT)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B78_voice_tracker_integration",
                     config=cfg_w, tags=["B78", "asap", "voice-tracker", "phase-d",
                                          "integration"],
                     dir="logs/wandb")

    if not CHECKPOINT.exists():
        print(f"FATAL: B76 checkpoint not at {CHECKPOINT}; run B76 first")
        run.finish(); return
    print(f"loading B76 voice tracker from {CHECKPOINT}")
    ckpt = torch.load(str(CHECKPOINT), map_location="cuda", weights_only=False)
    model = VoiceTransformer(d_model=ckpt["config"]["d_model"],
                              n_layers=ckpt["config"]["n_layers"]).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  best train acc: {ckpt['best_acc']:.4f}")
    print(f"  per-piece: {ckpt['per_piece']}")

    rows = []
    for piece, safe in PIECES.items():
        print(f"\n=== {piece} ===")
        try:
            notes = transcribe_via_cache(piece, backend="yourmt3plus")
        except FileNotFoundError as e:
            print(f"  skip: {e}")
            continue
        if not notes:
            print(f"  no notes from cache"); continue
        print(f"  {len(notes)} notes from cached YMT3+ transcription")

        ann = ASAP / piece / "midi_score_annotations.txt"
        if not ann.exists():
            print(f"  no annotations; skip"); continue
        beats = load_score_beats(ann)
        ref_iv, ref_pi = load_score_notes(piece)

        # Baseline: greedy adaptive_pj
        q_on_b, q_off_b = quantize_with_baseline_tracker(notes, beats)
        # Reconstruct intervals from tatum positions for snap measurement
        # (this is what gate_asap_rhythm does)
        # Actually for snap, we score the QUANTIZED intervals directly
        tatum_dur = (beats[-1] - beats[0]) / max(len(beats) - 1, 1) / 24
        base_iv = np.array([[beats[0] + q_on_b[i] * tatum_dur,
                              beats[0] + q_off_b[i] * tatum_dur]
                             for i in range(len(notes))])
        base_pi = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes])
        f1_base = score_snap(base_iv, base_pi, ref_iv, ref_pi)

        # B78: B76 transformer tracker
        q_on_t, q_off_t = quantize_with_b76_tracker(notes, beats, model)
        t_iv = np.array([[beats[0] + q_on_t[i] * tatum_dur,
                           beats[0] + q_off_t[i] * tatum_dur]
                          for i in range(len(notes))])
        f1_t = score_snap(t_iv, base_pi, ref_iv, ref_pi)

        delta = f1_t - f1_base
        rows.append({"piece": piece, "f1_baseline": f1_base, "f1_b78": f1_t,
                      "delta": delta, "n_notes": len(notes)})
        print(f"  baseline (greedy adaptive_pj): F1 = {f1_base:.4f}")
        print(f"  B78 (Transformer):             F1 = {f1_t:.4f}")
        print(f"  delta:                         {delta:+.4f}")
        wandb.log({"piece": piece, "f1_baseline": f1_base, "f1_b78": f1_t, "delta": delta})

    if not rows:
        run.finish(); return

    base_mean = float(np.mean([r["f1_baseline"] for r in rows]))
    b78_mean = float(np.mean([r["f1_b78"] for r in rows]))
    print(f"\nMEAN over {len(rows)} pieces:")
    print(f"  baseline: {base_mean:.4f}")
    print(f"  B78:      {b78_mean:.4f}")
    print(f"  delta:    {b78_mean - base_mean:+.4f}")
    wandb.summary.update({"mean_baseline_f1": base_mean, "mean_b78_f1": b78_mean,
                           "mean_delta": b78_mean - base_mean})
    OUT_JSON.write_text(json.dumps({"rows": rows, "mean_baseline_f1": base_mean,
                                     "mean_b78_f1": b78_mean,
                                     "mean_delta": b78_mean - base_mean,
                                     "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
