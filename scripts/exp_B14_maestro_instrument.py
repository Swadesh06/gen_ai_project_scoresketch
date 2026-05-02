"""Exp B14: full instrument pipeline (input_kind=piano) on short MAESTRO clips.
Renders MAESTRO MIDIs to wav, runs ByteDance + DP, scores against the MIDI."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

MAESTRO = Path("~/datasets/maestro").expanduser()
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"


def render_midi(midi: Path, wav: Path, sr: int = 22050) -> None:
    if wav.exists() and wav.stat().st_size > 0:
        return
    cmd = ["fluidsynth", "-ni", "-r", str(sr), "-F", str(wav), "-T", "wav", SF2, str(midi)]
    subprocess.run(cmd, check=True, capture_output=True)


def crop_midi(src: Path, dst: Path, max_seconds: float) -> Path:
    pm = pretty_midi.PrettyMIDI(str(src))
    keep_inst = []
    for inst in pm.instruments:
        notes = [n for n in inst.notes if n.start < max_seconds]
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in notes:
            new.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                               start=n.start, end=min(n.end, max_seconds)))
        if new.notes:
            keep_inst.append(new)
    out = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo() if pm.instruments else 120)
    out.instruments.extend(keep_inst)
    out.write(str(dst))
    return dst


def gt_from_midi(mid: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv: list[list[float]] = []; pi: list[float] = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv, dtype=np.float64), np.array(pi, dtype=np.float64)


def run_one(midi_path: Path, work_dir: Path, max_seconds: float) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    cropped = work_dir / f"{midi_path.stem}_30s.mid"
    crop_midi(midi_path, cropped, max_seconds)
    wav = cropped.with_suffix(".wav")
    render_midi(cropped, wav)
    notes = transcribe_piano(str(wav))
    onsets = np.array([n.onset_s for n in notes], dtype=np.float64)
    offsets = np.array([n.offset_s for n in notes], dtype=np.float64)
    beats, _, bpm = track_beats_beat_this(str(wav))
    if len(beats) >= 2 and len(notes) > 0:
        q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, beats, tatums_per_beat=24)
    gt_iv, gt_p = gt_from_midi(cropped)
    if len(notes) == 0 or len(gt_iv) == 0:
        return {"piece": midi_path.name, "n_pred": len(notes), "n_gt": len(gt_iv),
                "f1": 0.0, "p": 0.0, "r": 0.0, "bpm": float(bpm)}
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes], dtype=np.float64)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return {"piece": midi_path.name, "n_pred": int(len(notes)), "n_gt": int(len(gt_iv)),
            "f1": float(f), "p": float(p), "r": float(r), "bpm": float(bpm)}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(n_pieces: int, max_seconds: float) -> None:
    midis = sorted((MAESTRO / "2018").glob("*.midi"))[:n_pieces]
    work = Path("outputs/maestro_clips")
    cfg = {"exp": "B14_maestro_instrument", "n_pieces": len(midis),
            "max_seconds": max_seconds, "git_sha": git_sha()}
    run = wandb.init(
        project="humscribe-v3.2", name=f"exp_B14_maestro_n{len(midis)}",
        config=cfg, tags=["B14", "maestro", "instrument"], dir="logs/wandb",
    )
    rows: list[dict] = []
    for i, m in enumerate(midis):
        print(f"\n[{i+1}/{len(midis)}] {m.name}")
        try:
            r = run_one(m, work, max_seconds)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        rows.append(r)
        print(f"  P={r['p']:.3f}  R={r['r']:.3f}  F1={r['f1']:.3f}  pred={r['n_pred']}  gt={r['n_gt']}  bpm={r['bpm']:.0f}")
        wandb.log({"piece_idx": i, **{f"piece/{k}": v for k, v in r.items() if isinstance(v, (int, float))}})
    if rows:
        f1s = [r["f1"] for r in rows]; ps = [r["p"] for r in rows]; rs = [r["r"] for r in rows]
        summary = {"n": len(rows), "mean_f1": float(np.mean(f1s)),
                   "mean_p": float(np.mean(ps)), "mean_r": float(np.mean(rs))}
        wandb.summary.update(summary)
        print(f"\nSummary: mean F1={summary['mean_f1']:.3f}  P={summary['mean_p']:.3f}  R={summary['mean_r']:.3f}")
    out = Path("reports/_exp_B14_maestro_instrument.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary if rows else {}}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pieces", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    main(**vars(ap.parse_args()))
