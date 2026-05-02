"""Re-sweep voicing_threshold and min_note_seconds with the new psw=15 default."""
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


VOC = Path("~/datasets/vocadito").expanduser()


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


def main() -> None:
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B26_voc_resweep_psw15",
                     config={"git_sha": git_sha(), "psw": 15},
                     tags=["B26", "vocadito", "resweep"], dir="logs/wandb")
    rows = []
    for vt in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        for mns in (0.04, 0.05, 0.06, 0.08, 0.10):
            for oms in (0.02, 0.03, 0.05, 0.08):
                mc = ModeConfig(voicing_threshold=vt, min_note_seconds=mns,
                                onset_merge_seconds=oms, dp_offgrid_penalty=0.5,
                                pitch_smooth_window=15)
                f1s = []
                for nf in files:
                    cid = nf.stem.replace("_notesA1", "")
                    wav = audio_dir / f"{cid}.wav"
                    if not wav.exists():
                        continue
                    gt_iv, gt_p = load_notes(nf)
                    notes = predict(wav, mc)
                    _, _, f = score(notes, gt_iv, gt_p)
                    f1s.append(f)
                mf = float(np.mean(f1s))
                rows.append({"vt": vt, "mns": mns, "oms": oms, "f1": mf})
                wandb.log({"vt": vt, "mns": mns, "oms": oms, "f1": mf})

    rows.sort(key=lambda r: -r["f1"])
    print(f"\nTop 10 (psw=15):")
    for r in rows[:10]:
        print(f"  F1={r['f1']:.3f}  vt={r['vt']}  mns={r['mns']}  oms={r['oms']}")
    out = Path("reports/_exp_B26_voc_resweep.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    main()
