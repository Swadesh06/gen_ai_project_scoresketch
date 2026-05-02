"""basic_pitch (ICASSP 2022) on Vocadito — generic instrument transcriber on humming."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.instrument.basic_pitch import transcribe_basic_pitch
from humscribe.notes import NoteEvent, midi_to_hz


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def score(notes, iv, hz):
    if not notes:
        return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, hz, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return float(p), float(r), float(f)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(annotator: str = "A1") -> None:
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob(f"*_notes{annotator}.csv"))
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B34_basicpitch_voc_{annotator}",
                     config={"git_sha": git_sha(), "annotator": annotator},
                     tags=["B34", "vocadito", "basic_pitch"], dir="logs/wandb")
    rows = []
    for nf in files:
        cid = nf.stem.replace(f"_notes{annotator}", "")
        wav = audio_dir / f"{cid}.wav"
        if not wav.exists():
            continue
        try:
            notes = transcribe_basic_pitch(str(wav))
        except Exception as e:
            print(f"{cid}: failed -- {e}")
            continue
        gt_iv, gt_p = load_notes(nf)
        p, r, f = score(notes, gt_iv, gt_p)
        rows.append({"clip": cid, "p": p, "r": r, "f1": f, "n_pred": len(notes), "n_ref": len(gt_p)})
        print(f"{cid:20s}  P={p:.3f} R={r:.3f} F1={f:.3f}  pred={len(notes)} ref={len(gt_p)}")
        wandb.log({"clip": cid, "p": p, "r": r, "f1": f})
    if rows:
        f1s = [r["f1"] for r in rows]
        ps = [r["p"] for r in rows]
        rs = [r["r"] for r in rows]
        summary = {"mean_f1": float(np.mean(f1s)), "mean_p": float(np.mean(ps)), "mean_r": float(np.mean(rs)), "n": len(rows)}
        wandb.summary.update(summary)
        print(f"\nMean F1: {summary['mean_f1']:.3f}  P: {summary['mean_p']:.3f}  R: {summary['mean_r']:.3f}")
    out = Path(f"reports/_exp_B34_basicpitch_voc_{annotator}.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary if rows else {}}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    main()
