"""WandB-instrumented runner for the Stage 2-B.1 MIR-1K PESTO sanity gate.

Wraps the verbatim eval logic in scripts/eval_mir1k_pitch_sanity.py so we get
per-clip RPA + a mean to the WandB dashboard, plus a JSON record on disk.
Gate threshold per spec: mean RPA > 0.85 (slightly below PESTO's published 0.90
to account for voicing-window differences).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf
import wandb

from humscribe.pitch.pesto_track import track_pitch_pesto


def load_pv(pv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    midi = np.array([float(x) for x in pv_path.read_text().split()])
    times = np.arange(len(midi)) * 0.020 + 0.020
    return times, midi


def git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main(mir1k_dir: str, n_clips: int, seed: int) -> None:
    root = Path(mir1k_dir).expanduser()
    wavs = sorted((root / "Wavfile").glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs under {root/'Wavfile'}")
    rng = random.Random(seed)
    sample = rng.sample(wavs, min(n_clips, len(wavs)))

    cfg = {
        "gate": "mir1k_pitch_sanity",
        "n_clips": len(sample),
        "seed": seed,
        "mir1k_dir": str(root),
        "git_sha": git_sha(),
        "model": "pesto",
    }
    run = wandb.init(
        project="humscribe-v3.2",
        name=f"gate_mir1k_n{len(sample)}_seed{seed}",
        config=cfg,
        tags=["gate", "stage2b1", "pesto"],
        dir="logs/wandb",
    )

    rpas: list[float] = []
    per_clip: list[dict] = []
    for i, wav in enumerate(sample):
        audio, sr = sf.read(str(wav))
        if audio.ndim == 2:
            audio = audio[:, 1]
        gt_t, gt_midi = load_pv(root / "PitchLabel" / wav.with_suffix(".pv").name)
        pred_t, pred_hz, _conf = track_pitch_pesto(audio.astype(np.float32), sr)
        pred_cents = 1200 * np.log2(pred_hz / 440.0 + 1e-9) + 6900
        pred_at_gt = np.interp(gt_t, pred_t, pred_cents)
        gt_voicing = (gt_midi > 0).astype(bool)
        gt_cents = np.where(gt_voicing, gt_midi * 100, 0.0)
        rpa = mir_eval.melody.raw_pitch_accuracy(
            gt_voicing, gt_cents, gt_voicing, pred_at_gt, cent_tolerance=50,
        )
        rpa = float(rpa)
        rpas.append(rpa)
        per_clip.append({"clip": wav.name, "rpa": rpa, "n_frames": int(len(gt_t))})
        wandb.log({"clip_idx": i, "rpa": rpa})
        print(f"{wav.name:30s}  RPA={rpa:.3f}")

    mean = float(np.mean(rpas))
    p25, p50, p75 = (float(x) for x in np.percentile(rpas, [25, 50, 75]))
    pass_gate = mean > 0.85
    summary = {
        "mean_rpa": mean,
        "median_rpa": p50,
        "p25_rpa": p25,
        "p75_rpa": p75,
        "n": len(rpas),
        "gate_pass": pass_gate,
    }
    print(f"\nMean RPA: {mean:.3f}  N={len(rpas)}")
    print(f"GATE: {'PASS' if pass_gate else 'FAIL - fix loading/voicing, not PESTO'}")
    wandb.log(summary)
    wandb.summary.update(summary)

    out_path = Path("reports/_gate_mir1k.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=2))
    print(f"\nrun: {run.url}")
    print(f"json: {out_path}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mir1k-dir", default="~/datasets/mir1k/MIR-1K")
    ap.add_argument("--n-clips", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.mir1k_dir, args.n_clips, args.seed)
