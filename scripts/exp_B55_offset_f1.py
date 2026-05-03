"""B55: re-evaluate Vocadito under stricter Onset-Offset F1 (offset_ratio=0.2).
Currently we use offset_ratio=None (offset unconstrained). Adding the offset
constraint penalizes wrong durations — a different optimization target."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pipeline import transcribe_humming


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def score_pair(ref_iv, ref_p, est_iv, est_p, offset_ratio):
    if est_iv.shape[0] == 0: return 0.0, 0.0, 0.0
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=offset_ratio, offset_min_tolerance=0.05)
    return float(p), float(r), float(f)


def transcribe(wav: Path):
    audio, sr = load_audio(str(wav), target_sr=22050)
    cfg = PipelineConfig(mode="soft", input_kind="humming", pitch_model="pesto_crepevoicing")
    res = transcribe_humming(audio, sr, cfg=cfg)
    return res.notes


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B55_offset_f1",
                     config={"git_sha": git_sha()}, tags=["B55", "vocadito", "offset"], dir="logs/wandb")
    rows = []
    for nf in files:
        cid = nf.stem.replace("_notesA1", "")
        wav = audio_dir / f"{cid}.wav"
        if not wav.exists(): continue
        gt_iv, gt_p = load_notes(nf)
        notes = transcribe(wav)
        if not notes:
            rows.append({"clip": cid, "f_no_offset": 0.0, "f_offset20": 0.0, "f_offset50": 0.0})
            continue
        eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
        ep = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
        _, _, f0 = score_pair(gt_iv, gt_p, eiv, ep, None)
        _, _, f20 = score_pair(gt_iv, gt_p, eiv, ep, 0.2)
        _, _, f50 = score_pair(gt_iv, gt_p, eiv, ep, 0.5)
        rows.append({"clip": cid, "f_no_offset": f0, "f_offset20": f20, "f_offset50": f50})
        print(f"  {cid:25s}  no_off={f0:.3f}  off20={f20:.3f}  off50={f50:.3f}")
    if not rows:
        print("no data"); run.finish(); return
    means = {k: float(np.mean([r[k] for r in rows])) for k in ("f_no_offset", "f_offset20", "f_offset50")}
    print("\nMeans:")
    for k, v in means.items(): print(f"  {k:15s} = {v:.3f}")
    wandb.summary.update(means)
    out = Path("reports/_exp_B55_offset_f1.json")
    out.write_text(json.dumps({"rows": rows, "means": means}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
