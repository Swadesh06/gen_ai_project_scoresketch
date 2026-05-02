"""WandB-sweep-friendly runner for Vocadito humming-pipeline hyperparameters.

Designed to be the entry point of a `wandb sweep`. WandB injects the swept
config into `wandb.config`; we override the soft/medium/hard mode-config fields
before running per-clip evaluation.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig, PipelineConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes


def load_vocadito_notes(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in csv_path.read_text().splitlines() if r.strip()]
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durs = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([onsets, onsets + durs], axis=1), pitches


def predict_notes(audio_path: str, mc: ModeConfig, sr: int = 22050) -> list[NoteEvent]:
    audio, asr = load_audio(audio_path, target_sr=sr)
    t, hz, vc = track_pitch_pesto(audio, asr)
    notes = segment_pitch_to_notes(t, hz, vc, mc)
    return [n for n in notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]


def score_clip(notes: list[NoteEvent], gt_iv: np.ndarray, gt_hz: np.ndarray) -> dict:
    if not notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0, "n_ref": int(len(gt_hz))}
    est_iv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes], dtype=np.float64)
    est_hz = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes], dtype=np.float64)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_hz, est_iv, est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "n_pred": int(len(est_hz)), "n_ref": int(len(gt_hz))}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    run = wandb.init(
        project="humscribe-v3.2",
        tags=["sweep", "vocadito", "humming"],
        dir="logs/wandb",
    )
    cfg = wandb.config
    voc_dir = Path(cfg.get("vocadito_dir", "~/datasets/vocadito")).expanduser()
    annotator = cfg.get("annotator", "A1")
    notes_dir = voc_dir / "Annotations" / "Notes"
    audio_dir = voc_dir / "Audio"
    note_files = sorted(notes_dir.glob(f"*_notes{annotator}.csv"))

    mc = ModeConfig(
        voicing_threshold=float(cfg.voicing_threshold),
        min_note_seconds=float(cfg.min_note_seconds),
        onset_merge_seconds=float(cfg.get("onset_merge_seconds", 0.05)),
        dp_offgrid_penalty=float(cfg.get("dp_offgrid_penalty", 1.0)),
        pitch_smooth_window=int(cfg.pitch_smooth_window),
    )
    wandb.config.update({"git_sha": git_sha(), "annotator": annotator}, allow_val_change=True)

    f1s, ps, rs = [], [], []
    for nf in note_files:
        clip_id = nf.stem.replace(f"_notes{annotator}", "")
        wav = audio_dir / f"{clip_id}.wav"
        if not wav.exists():
            continue
        gt_iv, gt_hz = load_vocadito_notes(nf)
        notes = predict_notes(str(wav), mc)
        s = score_clip(notes, gt_iv, gt_hz)
        f1s.append(s["f1"]); ps.append(s["precision"]); rs.append(s["recall"])

    summary = {
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "median_f1": float(np.median(f1s)) if f1s else 0.0,
        "mean_p": float(np.mean(ps)) if ps else 0.0,
        "mean_r": float(np.mean(rs)) if rs else 0.0,
        "n": len(f1s),
    }
    wandb.summary.update(summary)
    print(json.dumps(summary))
    run.finish()


if __name__ == "__main__":
    main()
