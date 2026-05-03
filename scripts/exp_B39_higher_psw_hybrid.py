"""B39: even higher psw with hybrid voicing. B36b found psw=19 best in {11,15,19}; try {19, 21, 25, 31}."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def predict(wav: Path, mc: ModeConfig):
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, _pv = track_pitch_pesto(audio, sr)
    ct, _ch, cv = track_pitch_crepe(audio, sr)
    voicing = np.interp(pt, ct, cv)
    notes = segment_pitch_to_notes(pt, ph, voicing, mc)
    return [n for n in notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]


def score(notes, iv, hz):
    if not notes: return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, hz, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B39_higher_psw_hybrid",
                     config={"git_sha": git_sha()}, tags=["B39", "vocadito", "psw"], dir="logs/wandb")
    rows = []
    for psw in (19, 21, 23, 25, 27, 31, 35):
        for vt in (0.70, 0.75, 0.80):
            mc = ModeConfig(voicing_threshold=vt, min_note_seconds=0.052,
                            onset_merge_seconds=0.026, dp_offgrid_penalty=0.5,
                            pitch_smooth_window=psw)
            f1s = []
            for nf in files:
                cid = nf.stem.replace("_notesA1", "")
                wav = audio_dir / f"{cid}.wav"
                if not wav.exists(): continue
                gt_iv, gt_p = load_notes(nf)
                notes = predict(wav, mc)
                _, _, f = score(notes, gt_iv, gt_p)
                f1s.append(f)
            mf = float(np.mean(f1s))
            rows.append({"psw": psw, "vt": vt, "f1": mf})
            print(f"  psw={psw:2d} vt={vt:.2f}  F1={mf:.3f}")
            wandb.log({"psw": psw, "vt": vt, "f1": mf})
    rows.sort(key=lambda r: -r["f1"])
    print(f"\nTop 5:")
    for r in rows[:5]:
        print(f"  F1={r['f1']:.3f}  psw={r['psw']} vt={r['vt']}")
    out = Path("reports/_exp_B39_higher_psw.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
