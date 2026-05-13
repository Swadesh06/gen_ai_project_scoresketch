"""Phase G strict measurement: gate_vocadito_conp with G-4/5/6 toggleable.

Mirrors `scripts/gate_vocadito_conp.py`'s mir_eval transcription COnP
scoring (precision / recall / no-offset F1) but routes the pitch trace
through G-5 median smoothing before segmentation and then through G-4
same-pitch merging after segmentation. G-6 silent-region trimming
operates on the audio prior to anything else (it has no effect on
noff F1 unless the leading silence is bigger than the segmenter's
min-note threshold, which it usually isn't).

Two modes:
  --phase-g-post off   = vanilla baseline (matches the existing gate)
  --phase-g-post on    = G-4 + G-5 + G-6 applied at the production defaults

Writes the per-clip + summary to
`reports/_gate_vocadito_phase_g_{on|off}_A{1,2}.json`.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.post_process import (
    median_smooth_pitch, merge_same_pitch, trim_silence,
)


def _load_vocadito_notes(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in csv_path.read_text().splitlines() if r.strip()]
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitch_hz = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    intervals = np.stack([onsets, onsets + durations], axis=1)
    return intervals, pitch_hz


def predict_notes(audio_path: str, mode: str = "soft",
                   pitch_model: str = "pesto_crepevoicing",
                   apply_g4: bool = False, apply_g5: bool = False) -> list[NoteEvent]:
    cfg = PipelineConfig(input_kind="humming", mode=mode, pitch_model=pitch_model)
    audio, sr = load_audio(audio_path, target_sr=cfg.sample_rate)
    # NOTE: G-6 silent_trim is intentionally NOT applied here. trim_silence
    # truncates the audio AND shifts time-zero — but the GT note onsets are
    # absolute times in the ORIGINAL audio. If we trim and don't shift
    # predicted onsets back by `lead_s`, the noff F1 collapses (predicted
    # onsets land in the GT's silent prefix). The production `pipeline.py`
    # only routes the trimmed audio into `beat_this` (which is irrelevant
    # to this gate, which scores notes-only), and keeps the original audio
    # for segmentation.
    if pitch_model == "pesto_crepevoicing":
        t, hz, _pv = track_pitch_pesto(audio, sr)
        ct, _ch, cv = track_pitch_crepe(audio, sr)
        vc = np.interp(t, ct, cv) if len(ct) > 0 else _pv
    elif pitch_model == "pesto":
        t, hz, vc = track_pitch_pesto(audio, sr)
    elif pitch_model == "crepe":
        t, hz, vc = track_pitch_crepe(audio, sr)
    else:
        raise ValueError(f"unknown pitch_model: {pitch_model!r}")
    if apply_g5:
        # G-5: 250 ms voiced-only median smoothing on the pitch trace.
        hz, vc = median_smooth_pitch(t, hz, vc, window_ms=cfg.median_smooth_window_ms)
    notes = segment_pitch_to_notes(t, hz, vc, cfg.mode_config)
    notes = [n for n in notes if (n.offset_s - n.onset_s) >= cfg.mode_config.min_note_seconds]
    if apply_g4:
        # G-4: same-pitch gap merging.
        notes = merge_same_pitch(notes, gap_s=cfg.same_pitch_merge_ms / 1000.0)
    return notes


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocadito-dir", default="/workspace/.cache/vocadito_orig")
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--mode", default="soft")
    ap.add_argument("--pitch-model", default="pesto_crepevoicing")
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--pitch-tol-cents", type=float, default=50.0)
    ap.add_argument("--phase-g-post", choices=["on", "off"], default="on")
    ap.add_argument("--apply", default=None,
                     help="comma-separated subset of {g4,g5}: overrides --phase-g-post")
    ap.add_argument("--n-clips", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.apply is not None:
        sel = set(args.apply.split(",")) if args.apply else set()
        apply_g4 = "g4" in sel
        apply_g5 = "g5" in sel
        tag = args.apply or "none"
    else:
        apply_g4 = apply_g5 = (args.phase_g_post == "on")
        tag = args.phase_g_post
    root = Path(args.vocadito_dir).expanduser()
    audio_dir = root / "Audio"
    notes_dir = root / "Annotations" / "Notes"
    pat = f"_notes{args.annotator}.csv"
    note_files = sorted(p for p in notes_dir.glob(f"*{pat}"))
    if args.n_clips > 0:
        note_files = note_files[: args.n_clips]
    if not note_files:
        raise SystemExit(f"no annotations matching *{pat} under {notes_dir}")
    per_clip = []
    for nf in note_files:
        clip_id = nf.stem.replace(f"_notes{args.annotator}", "")
        wav = audio_dir / f"{clip_id}.wav"
        if not wav.exists():
            print(f"skip {clip_id}: missing audio")
            continue
        gt_iv, gt_hz = _load_vocadito_notes(nf)
        try:
            notes = predict_notes(str(wav), args.mode, args.pitch_model,
                                    apply_g4=apply_g4, apply_g5=apply_g5)
            sc = score_clip(notes, gt_iv, gt_hz, args.onset_tol, args.pitch_tol_cents)
        except Exception as e:
            print(f"skip {clip_id}: {e}")
            continue
        sc["clip"] = clip_id
        per_clip.append(sc)
        print(f"{clip_id:18s} P={sc['precision']:.3f} R={sc['recall']:.3f} F1={sc['f1']:.3f} pred={sc['n_pred']} ref={sc['n_ref']}")
    if not per_clip:
        raise SystemExit("no clips scored")
    f1s = [c["f1"] for c in per_clip]
    summary = {
        "mean_f1": float(np.mean(f1s)),
        "median_f1": float(np.median(f1s)),
        "mean_p": float(np.mean([c["precision"] for c in per_clip])),
        "mean_r": float(np.mean([c["recall"] for c in per_clip])),
        "n_clips": len(per_clip),
        "apply": tag,
        "apply_g4": apply_g4,
        "apply_g5": apply_g5,
        "annotator": args.annotator,
        "pitch_model": args.pitch_model,
        "onset_tol": args.onset_tol,
        "pitch_tol_cents": args.pitch_tol_cents,
    }
    out_path = (Path(args.out) if args.out
                else Path(f"reports/_gate_vocadito_phase_g_{tag}_{args.annotator}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=2))
    print(f"\nMean F1 ({tag}, {args.annotator}, n={len(per_clip)}): {summary['mean_f1']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
