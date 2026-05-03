"""B57: sweep onset_merge_seconds + voicing_threshold to fix offset accuracy.
Hypothesis: vibrato dips break notes mid-vibrato, fragmenting durations.
Increasing oms (merge close voiced segments) might fix this without harming onsets.
Target: lift Vocadito offset20 F1 from 0.439 toward IAA ceiling 0.642."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig, ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def predict(wav: Path, vt: float, oms: float, mns: float, psw: int):
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, _pv = track_pitch_pesto(audio, sr)
    ct, _ch, cv = track_pitch_crepe(audio, sr)
    voicing = np.interp(pt, ct, cv)
    mc = ModeConfig(voicing_threshold=vt, min_note_seconds=mns,
                     onset_merge_seconds=oms, dp_offgrid_penalty=0.5, pitch_smooth_window=psw)
    return segment_pitch_to_notes(pt, ph, voicing, mc)


def score(notes, gt_iv, gt_p, offset_ratio):
    if not notes: return 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    ep = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, eiv, ep, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=offset_ratio, offset_min_tolerance=0.05)
    return float(f)


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B57_oms_sweep",
                     config={"git_sha": git_sha()}, tags=["B57", "vocadito", "offset"], dir="logs/wandb")
    # Default: vt=0.75, oms=0.026, mns=0.052, psw=19
    # Sweep oms and mns since vt+psw are well-tuned for no_offset
    grid = []
    for vt in (0.65, 0.75):
        for oms in (0.026, 0.05, 0.1, 0.15, 0.2):
            for mns in (0.052, 0.1, 0.15):
                grid.append((vt, oms, mns, 19))
    print(f"{len(grid)} configs to test on {len(files)} clips")
    rows = []
    for vt, oms, mns, psw in grid:
        no_offs = []; off20s = []; off50s = []
        for nf in files:
            cid = nf.stem.replace("_notesA1", "")
            wav = audio_dir / f"{cid}.wav"
            if not wav.exists(): continue
            gt_iv, gt_p = load_notes(nf)
            notes = predict(wav, vt, oms, mns, psw)
            no_offs.append(score(notes, gt_iv, gt_p, None))
            off20s.append(score(notes, gt_iv, gt_p, 0.2))
            off50s.append(score(notes, gt_iv, gt_p, 0.5))
        f0 = float(np.mean(no_offs)); f20 = float(np.mean(off20s)); f50 = float(np.mean(off50s))
        rows.append({"vt": vt, "oms": oms, "mns": mns, "psw": psw, "no_off": f0, "off20": f20, "off50": f50})
        print(f"  vt={vt} oms={oms:.3f} mns={mns:.3f} psw={psw}  no={f0:.3f} o20={f20:.3f} o50={f50:.3f}")
        wandb.log({"vt": vt, "oms": oms, "mns": mns, "no_off": f0, "off20": f20, "off50": f50})
    rows.sort(key=lambda r: -r["off20"])
    print(f"\nTop 5 by off20:")
    for r in rows[:5]:
        print(f"  o20={r['off20']:.3f}  o50={r['off50']:.3f}  no={r['no_off']:.3f}  vt={r['vt']} oms={r['oms']} mns={r['mns']}")
    out = Path("reports/_exp_B57_oms_sweep.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
