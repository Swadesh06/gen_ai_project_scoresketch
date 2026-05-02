"""Vocadito quantitative gate (Phase 2 primary). For each clip, run the humming
pipeline through Stage 3 (notes) — bypassing rhythm quantization and beat
tracking, since the gate compares against absolute-time note annotations — and
score with `mir_eval.transcription.precision_recall_f1_overlap` in COnP mode
(onset + pitch only, ignoring offsets).

Per-clip and aggregate F1 logged to WandB. Default soft mode; can sweep modes
with --modes.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.ensemble import track_pitch_ensemble
from humscribe.pitch.hmm_segment import segment_pitch_to_notes_hmm
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.notes import NoteEvent, midi_to_hz


def load_vocadito_notes(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in csv_path.read_text().splitlines() if r.strip()]
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitch_hz = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    intervals = np.stack([onsets, onsets + durations], axis=1)
    return intervals, pitch_hz


def predict_notes(audio_path: str, mode: str, pitch_model: str = "pesto",
                  segmenter: str = "voicing") -> list[NoteEvent]:
    cfg = PipelineConfig(
        input_kind="humming", mode=mode,
        pitch_model=pitch_model, note_segmenter=segmenter,
    )
    audio, sr = load_audio(audio_path, target_sr=cfg.sample_rate)
    if pitch_model == "pesto":
        t, hz, vc = track_pitch_pesto(audio, sr)
    elif pitch_model == "crepe":
        t, hz, vc = track_pitch_crepe(audio, sr)
    elif pitch_model == "ensemble":
        t, hz, vc = track_pitch_ensemble(audio, sr)
    else:
        raise ValueError(f"unknown pitch_model: {pitch_model!r}")
    if segmenter == "voicing":
        notes = segment_pitch_to_notes(t, hz, vc, cfg.mode_config)
    elif segmenter == "hmm":
        notes = segment_pitch_to_notes_hmm(t, hz, vc, cfg.mode_config)
    else:
        raise ValueError(f"unknown segmenter: {segmenter!r}")
    return [n for n in notes if (n.offset_s - n.onset_s) >= cfg.mode_config.min_note_seconds]


def score_clip(notes: list[NoteEvent], gt_intervals: np.ndarray, gt_hz: np.ndarray,
               onset_tol: float, pitch_tol_cents: float) -> dict:
    if not notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0, "n_ref": int(len(gt_hz))}
    est_intervals = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes], dtype=np.float64)
    est_hz = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes], dtype=np.float64)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, gt_hz, est_intervals, est_hz,
        onset_tolerance=onset_tol,
        pitch_tolerance=pitch_tol_cents,
        offset_ratio=None,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "n_pred": int(len(est_hz)), "n_ref": int(len(gt_hz))}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(vocadito_dir: str, mode: str, annotator: str, n_clips: int,
         onset_tol: float, pitch_tol_cents: float, gate_threshold: float,
         pitch_model: str, segmenter: str) -> None:
    root = Path(vocadito_dir).expanduser()
    audio_dir = root / "Audio"
    notes_dir = root / "Annotations" / "Notes"
    pat = f"_notes{annotator}.csv"
    note_files = sorted(p for p in notes_dir.glob(f"*{pat}"))
    if n_clips > 0:
        note_files = note_files[:n_clips]
    if not note_files:
        raise SystemExit(f"no annotations matching *{pat} under {notes_dir}")

    cfg = {
        "gate": "vocadito_conp",
        "mode": mode,
        "annotator": annotator,
        "n_clips": len(note_files),
        "onset_tol": onset_tol,
        "pitch_tol_cents": pitch_tol_cents,
        "gate_threshold_f1": gate_threshold,
        "git_sha": git_sha(),
        "pitch_model": pitch_model,
        "segmenter": segmenter,
    }
    run = wandb.init(
        project="humscribe-v3.2",
        name=f"gate_vocadito_{mode}_{pitch_model}_{segmenter}_{annotator}_n{len(note_files)}",
        config=cfg,
        tags=["gate", "vocadito", "conp", f"mode-{mode}", f"pitch-{pitch_model}", f"seg-{segmenter}"],
        dir="logs/wandb",
    )

    per_clip = []
    for i, nf in enumerate(note_files):
        clip_id = nf.stem.replace(f"_notes{annotator}", "")
        wav = audio_dir / f"{clip_id}.wav"
        if not wav.exists():
            print(f"skip {clip_id}: missing audio")
            continue
        gt_iv, gt_hz = load_vocadito_notes(nf)
        notes = predict_notes(str(wav), mode, pitch_model, segmenter)
        scores = score_clip(notes, gt_iv, gt_hz, onset_tol, pitch_tol_cents)
        scores["clip"] = clip_id
        per_clip.append(scores)
        wandb.log({"clip_idx": i, **{f"clip/{k}": v for k, v in scores.items() if k != "clip"}})
        print(f"{clip_id:18s}  P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}  pred={scores['n_pred']}  ref={scores['n_ref']}")

    f1s = [c["f1"] for c in per_clip]
    ps = [c["precision"] for c in per_clip]
    rs = [c["recall"] for c in per_clip]
    summary = {
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "median_f1": float(np.median(f1s)) if f1s else 0.0,
        "mean_p": float(np.mean(ps)) if ps else 0.0,
        "mean_r": float(np.mean(rs)) if rs else 0.0,
        "n": len(per_clip),
        "gate_pass": (float(np.mean(f1s)) if f1s else 0.0) >= gate_threshold,
    }
    print(f"\nMean F1: {summary['mean_f1']:.3f}  (gate: >= {gate_threshold:.2f})")
    print(f"GATE: {'PASS' if summary['gate_pass'] else 'FAIL'}")
    wandb.log(summary)
    wandb.summary.update(summary)

    out = Path(f"reports/_gate_vocadito_{mode}_{pitch_model}_{segmenter}_{annotator}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_clip": per_clip, "config": cfg}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocadito-dir", default="~/datasets/vocadito")
    ap.add_argument("--mode", choices=["soft", "medium", "hard"], default="soft")
    ap.add_argument("--annotator", default="A1", help="A1 or A2")
    ap.add_argument("--n-clips", type=int, default=0, help="0 = all")
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--pitch-tol-cents", type=float, default=50.0)
    ap.add_argument("--gate-threshold", type=float, default=0.40,
                    help="F1 floor; v3.2 spec doesn't fix this — 0.40 is a 'not broken' bar")
    ap.add_argument("--pitch-model", choices=["pesto", "crepe", "ensemble"], default="pesto")
    ap.add_argument("--segmenter", choices=["voicing", "hmm"], default="voicing")
    args = ap.parse_args()
    main(args.vocadito_dir, args.mode, args.annotator, args.n_clips,
         args.onset_tol, args.pitch_tol_cents, args.gate_threshold,
         args.pitch_model, args.segmenter)
