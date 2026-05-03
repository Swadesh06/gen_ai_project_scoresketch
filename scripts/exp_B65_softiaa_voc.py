"""B65 — Soft-IAA Vocadito scoring (Phase C, B51 follow-up).

The Vocadito IAA ceiling (B51) was 0.740 no-offset. Standard practice has
been to score against A1 OR A2 separately. The "soft-IAA" framing scores
each prediction against both annotators and averages — same prediction,
two ground truths, mean F1.

Why this matters: with a single annotator the metric over-rewards mimicking
that annotator's idiosyncrasies. Soft-IAA is closer to "the answer
real-world humans would accept" since it tolerates either annotator's
style.

This is a pure metric change — no model or pipeline changes. Just re-scores
the existing predictions against both annotators, averages.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import midi_to_hz
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes

VOC = Path("~/datasets/vocadito").expanduser()
OUT_JSON = Path("reports/_exp_B65_softiaa_voc.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_notes(annotation_csv: Path):
    """Vocadito annotations are CSV: onset_s, pitch_hz, duration_s.
    Returns (intervals shape (n,2) in seconds, pitches shape (n,) in Hz).
    """
    rows = [r.strip().split(",") for r in annotation_csv.read_text().splitlines() if r.strip()]
    if not rows:
        return np.empty((0, 2)), np.empty(0)
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    intervals = np.stack([onsets, onsets + durations], axis=1)
    return intervals, pitches


def score(pred_intervals, pred_pitches, ref_intervals, ref_pitches, offset):
    if len(pred_intervals) == 0 or len(ref_intervals) == 0:
        return 0.0, 0.0, 0.0
    p, r, f = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches,
        offset_ratio=offset, onset_tolerance=0.05,
    )[:3]
    return p, r, f


def predict_clip(audio_path: str):
    """Same recipe as gate_vocadito_conp.py — pitch + voicing only, no DP.
    DP would shift onsets by tens of ms which collapses mir_eval's 50ms
    onset tolerance to zero matches."""
    audio, sr = load_audio(audio_path, target_sr=22050)
    t, hz, vc = track_pitch_hybrid_voicing(audio, sr)
    mode_cfg = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    notes = segment_pitch_to_notes(t, hz, vc, mode_cfg)
    notes = [n for n in notes if (n.offset_s - n.onset_s) >= mode_cfg.min_note_seconds]
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)]
                           for n in notes], dtype=np.float64)
    pitches = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi())
                         for n in notes], dtype=np.float64)
    return intervals, pitches


def main(n_clips: int = 40) -> None:
    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    clips = sorted(p.stem for p in audio_dir.glob("vocadito_*.wav"))[:n_clips]
    print(f"scoring {len(clips)} clips against A1 and A2 + soft-IAA average")
    cfg_w = {"git_sha": git_sha(), "n_clips": len(clips)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B65_softiaa_voc",
                     config=cfg_w, tags=["B65", "vocadito", "soft-iaa", "phase-c"],
                     dir="logs/wandb")
    rows = []
    for clip in clips:
        wav = audio_dir / f"{clip}.wav"
        a1 = ann_dir / f"{clip}_notesA1.csv"
        a2 = ann_dir / f"{clip}_notesA2.csv"
        if not (a1.exists() and a2.exists()):
            print(f"skip {clip}: missing annotation")
            continue
        try:
            pred_iv, pred_pi = predict_clip(str(wav))
        except Exception as e:
            print(f"  {clip} predict failed: {e}")
            continue
        ref1_iv, p1_hz = load_notes(a1)
        ref2_iv, p2_hz = load_notes(a2)
        out = {"clip": clip}
        for off_label, off_val in [("noff", None), ("o50", 0.5), ("o20", 0.2)]:
            f1_a1 = score(pred_iv, pred_pi, ref1_iv, p1_hz, off_val)[2]
            f1_a2 = score(pred_iv, pred_pi, ref2_iv, p2_hz, off_val)[2]
            out[f"{off_label}_a1"] = f1_a1
            out[f"{off_label}_a2"] = f1_a2
            out[f"{off_label}_soft"] = 0.5 * (f1_a1 + f1_a2)
        rows.append(out)
        print(f"  {clip}: noff a1={out['noff_a1']:.3f} a2={out['noff_a2']:.3f} soft={out['noff_soft']:.3f}")
    means = {k: float(np.mean([r[k] for r in rows]))
             for k in rows[0].keys() if k != "clip"}
    print("\nMEANS:")
    for k, v in means.items():
        print(f"  {k:12s} = {v:.4f}")
    wandb.summary.update(means)
    OUT_JSON.write_text(json.dumps({"rows": rows, "means": means, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
