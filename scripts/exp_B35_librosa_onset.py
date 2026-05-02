"""Use librosa.onset_detect to drive segmentation, then PESTO for pitch within each segment."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.pesto_track import track_pitch_pesto


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def predict(wav: Path, mc: ModeConfig, librosa_delta: float, librosa_pre_max: int) -> list[NoteEvent]:
    audio, sr = load_audio(str(wav), target_sr=22050)
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, units="time", backtrack=False,
        delta=librosa_delta, pre_max=librosa_pre_max, post_max=10,
        pre_avg=10, post_avg=10, wait=5,
    )
    if len(onset_frames) == 0:
        return []
    pt, ph, pv = track_pitch_pesto(audio, sr)
    end_t = float(pt[-1]) if len(pt) > 0 else len(audio) / sr
    onset_times = np.append(onset_frames, end_t)
    notes: list[NoteEvent] = []
    for k in range(len(onset_times) - 1):
        s_t = float(onset_times[k]); e_t = float(onset_times[k + 1])
        if (e_t - s_t) < mc.min_note_seconds:
            continue
        idx = np.where((pt >= s_t) & (pt < e_t))[0]
        if len(idx) == 0:
            continue
        seg_v = pv[idx]
        if seg_v.mean() < mc.voicing_threshold * 0.6:
            continue
        seg_h = ph[idx]
        valid = (seg_h > 0) & (pv[idx] >= mc.voicing_threshold * 0.5)
        if not valid.any():
            continue
        midi = float(np.median([_hz_to_midi(float(h)) for h in seg_h[valid]]))
        midi_int = int(round(midi)) if midi > 0 else 0
        notes.append(NoteEvent(
            onset_s=s_t, offset_s=e_t,
            pitch_hz=midi_to_hz(midi), pitch_midi=midi_int,
            confidence=float(seg_v.mean()),
        ))
    return notes


def _hz_to_midi(h: float) -> float:
    if h <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(h / 440.0)


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
    run = wandb.init(project="humscribe-v3.2", name="exp_B35_librosa_onset",
                     config={"git_sha": git_sha()},
                     tags=["B35", "vocadito", "librosa"], dir="logs/wandb")
    rows = []
    base = ModeConfig.for_mode("soft")
    for delta in (0.05, 0.1, 0.2, 0.3, 0.5):
        for pre_max in (3, 5, 10):
            f1s = []
            for nf in files:
                cid = nf.stem.replace("_notesA1", "")
                wav = audio_dir / f"{cid}.wav"
                if not wav.exists():
                    continue
                gt_iv, gt_p = load_notes(nf)
                notes = predict(wav, base, delta, pre_max)
                _, _, f = score(notes, gt_iv, gt_p)
                f1s.append(f)
            mf = float(np.mean(f1s))
            rows.append({"delta": delta, "pre_max": pre_max, "f1": mf})
            print(f"  delta={delta:.2f}  pre_max={pre_max:2d}  F1={mf:.3f}")
            wandb.log({"delta": delta, "pre_max": pre_max, "f1": mf})
    rows.sort(key=lambda r: -r["f1"])
    print(f"\nTop 3:")
    for r in rows[:3]:
        print(f"  F1={r['f1']:.3f} delta={r['delta']} pre_max={r['pre_max']}")
    out = Path("reports/_exp_B35_librosa.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    main()
