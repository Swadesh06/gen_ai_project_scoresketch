"""Run Vocadito gate over a 2x3 (annotator x mode) matrix and print a table."""
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
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC_DIR = Path("~/datasets/vocadito").expanduser()
ANNS = ("A1", "A2")
MODES = ("soft", "medium", "hard")


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def predict(wav: Path, mc: ModeConfig) -> list[NoteEvent]:
    audio, sr = load_audio(str(wav), target_sr=22050)
    t, hz, vc = track_pitch_pesto(audio, sr)
    notes = segment_pitch_to_notes(t, hz, vc, mc)
    return [n for n in notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]


def score(notes: list[NoteEvent], iv: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    if not notes:
        return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    pp, rr, ff, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, p, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return float(pp), float(rr), float(ff)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    audio_dir = VOC_DIR / "Audio"
    notes_dir = VOC_DIR / "Annotations" / "Notes"
    cfg = {"gate": "vocadito_matrix", "git_sha": git_sha()}
    run = wandb.init(
        project="humscribe-v3.2",
        name="gate_vocadito_matrix",
        config=cfg,
        tags=["gate", "vocadito", "matrix", "2x3"],
        dir="logs/wandb",
    )
    rows: dict[tuple[str, str], dict] = {}
    for ann in ANNS:
        files = sorted(notes_dir.glob(f"*_notes{ann}.csv"))
        for mode in MODES:
            mc = ModeConfig.for_mode(mode)
            f1s, ps, rs = [], [], []
            for nf in files:
                clip = nf.stem.replace(f"_notes{ann}", "")
                wav = audio_dir / f"{clip}.wav"
                if not wav.exists():
                    continue
                gt_iv, gt_p = load_notes(nf)
                notes = predict(wav, mc)
                pp, rr, ff = score(notes, gt_iv, gt_p)
                f1s.append(ff); ps.append(pp); rs.append(rr)
            row = {
                "mean_f1": float(np.mean(f1s)),
                "mean_p": float(np.mean(ps)),
                "mean_r": float(np.mean(rs)),
                "n": len(f1s),
            }
            rows[(ann, mode)] = row
            wandb.log({f"{ann}/{mode}/mean_f1": row["mean_f1"],
                       f"{ann}/{mode}/mean_p": row["mean_p"],
                       f"{ann}/{mode}/mean_r": row["mean_r"]})
            print(f"{ann}/{mode:6s}  F1={row['mean_f1']:.3f}  P={row['mean_p']:.3f}  R={row['mean_r']:.3f}  N={row['n']}")
    print("\n        " + " | ".join(f"{m:>15s}" for m in MODES))
    for ann in ANNS:
        line = [f"{ann:6s}"]
        for mode in MODES:
            r = rows[(ann, mode)]
            line.append(f"F1={r['mean_f1']:.3f}/P={r['mean_p']:.2f}/R={r['mean_r']:.2f}")
        print("  ".join(line))
    out = Path("reports/_gate_vocadito_matrix.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({f"{a}_{m}": v for (a, m), v in rows.items()}, indent=2))
    wandb.summary.update({f"{a}_{m}_F1": v["mean_f1"] for (a, m), v in rows.items()})
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    main()
