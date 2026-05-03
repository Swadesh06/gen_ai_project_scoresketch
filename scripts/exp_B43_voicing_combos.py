"""B43: try various ways to combine PESTO and CREPE voicing.
- pesto only (B22 baseline)
- crepe only (B36)
- (pesto + crepe) / 2
- max(pesto, crepe)
- min(pesto, crepe)
- pesto * crepe (geometric blend)
"""
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


def predict(wav: Path, mc: ModeConfig, combine: str):
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, pv_pesto = track_pitch_pesto(audio, sr)
    ct, _ch, cv_crepe_raw = track_pitch_crepe(audio, sr)
    cv = np.interp(pt, ct, cv_crepe_raw)
    if combine == "pesto":
        voicing = pv_pesto
    elif combine == "crepe":
        voicing = cv
    elif combine == "avg":
        voicing = (pv_pesto + cv) / 2
    elif combine == "max":
        voicing = np.maximum(pv_pesto, cv)
    elif combine == "min":
        voicing = np.minimum(pv_pesto, cv)
    elif combine == "geo":
        voicing = np.sqrt(pv_pesto * cv)
    else:
        raise ValueError(combine)
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


# Per-combine optimal vt (sensible starting points based on B22/B36b)
COMBINE_VT = {
    "pesto": 0.315, "crepe": 0.75, "avg": 0.50,
    "max": 0.65, "min": 0.20, "geo": 0.45,
}


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B43_voicing_combos",
                     config={"git_sha": git_sha()}, tags=["B43", "vocadito", "combine"], dir="logs/wandb")
    rows = []
    for combine, vt in COMBINE_VT.items():
        for vt_offset in (-0.10, -0.05, 0.0, 0.05, 0.10):
            this_vt = vt + vt_offset
            mc = ModeConfig(voicing_threshold=this_vt, min_note_seconds=0.052,
                            onset_merge_seconds=0.026, dp_offgrid_penalty=0.5,
                            pitch_smooth_window=19)
            f1s = []
            for nf in files:
                cid = nf.stem.replace("_notesA1", "")
                wav = audio_dir / f"{cid}.wav"
                if not wav.exists(): continue
                gt_iv, gt_p = load_notes(nf)
                notes = predict(wav, mc, combine)
                _, _, f = score(notes, gt_iv, gt_p)
                f1s.append(f)
            mf = float(np.mean(f1s))
            rows.append({"combine": combine, "vt": this_vt, "f1": mf})
            print(f"  {combine:6s} vt={this_vt:.2f}  F1={mf:.3f}")
            wandb.log({"combine": combine, "vt": this_vt, "f1": mf})
    rows.sort(key=lambda r: -r["f1"])
    print(f"\nTop 5:")
    for r in rows[:5]:
        print(f"  F1={r['f1']:.3f}  {r['combine']} vt={r['vt']:.2f}")
    out = Path("reports/_exp_B43_voicing_combos.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
