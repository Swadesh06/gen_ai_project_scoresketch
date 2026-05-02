"""WandB-sweep runner for the HMM segmenter on Vocadito (Exp B6)."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.hmm_segment import HMMConfig, segment_pitch_to_notes_hmm
from humscribe.pitch.pesto_track import track_pitch_pesto


def load_vocadito_notes(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in csv_path.read_text().splitlines() if r.strip()]
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durs = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([onsets, onsets + durs], axis=1), pitches


def score_clip(notes: list[NoteEvent], gt_iv: np.ndarray, gt_hz: np.ndarray) -> dict:
    if not notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    est_iv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes], dtype=np.float64)
    est_hz = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes], dtype=np.float64)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_hz, est_iv, est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    run = wandb.init(
        project="humscribe-v3.2",
        tags=["sweep", "vocadito", "hmm"],
        dir="logs/wandb",
    )
    cfg = wandb.config
    voc_dir = Path(cfg.get("vocadito_dir", "~/datasets/vocadito")).expanduser()
    annotator = cfg.get("annotator", "A1")
    notes_dir = voc_dir / "Annotations" / "Notes"
    audio_dir = voc_dir / "Audio"
    note_files = sorted(notes_dir.glob(f"*_notes{annotator}.csv"))

    hmm = HMMConfig(
        midi_lo=int(cfg.get("midi_lo", 36)),
        midi_hi=int(cfg.get("midi_hi", 96)),
        p_sustain=float(cfg.p_sustain),
        p_end=float(cfg.p_end),
        p_start=float(cfg.p_start),
        sigma_voicing=float(cfg.sigma_voicing),
        sigma_midi=float(cfg.sigma_midi),
        interval_decay=float(cfg.interval_decay),
    )
    mc = ModeConfig.for_mode("soft")
    wandb.config.update({"git_sha": git_sha()}, allow_val_change=True)

    f1s, ps, rs = [], [], []
    for nf in note_files:
        clip_id = nf.stem.replace(f"_notes{annotator}", "")
        wav = audio_dir / f"{clip_id}.wav"
        if not wav.exists():
            continue
        gt_iv, gt_hz = load_vocadito_notes(nf)
        audio, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_pesto(audio, sr)
        notes = segment_pitch_to_notes_hmm(t, hz, vc, mc, hmm)
        notes = [n for n in notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]
        s = score_clip(notes, gt_iv, gt_hz)
        f1s.append(s["f1"]); ps.append(s["precision"]); rs.append(s["recall"])

    summary = {
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_p": float(np.mean(ps)) if ps else 0.0,
        "mean_r": float(np.mean(rs)) if rs else 0.0,
        "n": len(f1s),
    }
    wandb.summary.update(summary)
    print(json.dumps(summary))
    run.finish()


if __name__ == "__main__":
    main()
