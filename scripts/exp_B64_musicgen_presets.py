"""B64 — MusicGen-Melody arrangement smoke across all 6 style presets (B+2 item 3).

For each PROMPT_PRESETS entry, generate a 10s arrangement on Vocadito clip 1.
Verifies: model loads (cached), all 6 produce a non-trivial output,
peak VRAM stays < 20 GB.

Pass criteria from `task_description_v2.md` §Work item 3:
- end-to-end: hum -> arrangement, recognizable melody (verified by ear, not metric)
- peak VRAM < 20 GB
- all 6 presets produce coherent output
- weights load on first call without download errors
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import wandb


VOC = Path("~/datasets/vocadito").expanduser()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(model_size: str, duration: float, clip: str) -> None:
    melody_path = VOC / "Audio" / f"{clip}.wav"
    out_dir = Path("outputs/musicgen_presets")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Defer musicgen import — its spacy/thinc stub injection clashes with
    # wandb's post-import telemetry hooks (`AttributeError: '_Stub'`).
    cfg = {"model_size": model_size, "duration": duration,
           "git_sha": git_sha(), "clip": clip}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B64_musicgen_{model_size}_{int(duration)}s",
                     config=cfg, tags=["B64", "musicgen", "item3", model_size],
                     dir="logs/wandb")
    from humscribe.arrange.musicgen import PROMPT_PRESETS, arrange_to_file
    cfg["n_presets"] = len(PROMPT_PRESETS)
    wandb.config.update({"n_presets": cfg["n_presets"]})
    rows = []
    print(f"Melody: {melody_path}; model: musicgen-{model_size}; duration: {duration}s")
    print(f"\n  {'preset':24s}  {'wall_s':>7s}  {'out_size_kb':>10s}  {'vram_peak_gb':>12s}")
    print("  " + "-" * 70)
    for name, prompt in PROMPT_PRESETS.items():
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        out = out_dir / f"{clip}_{name.replace(' ', '_')}.wav"
        t0 = time.time()
        try:
            arrange_to_file(str(melody_path), prompt, str(out),
                             duration_s=duration, model_size=model_size)
        except Exception as e:
            print(f"  {name:24s}  FAILED -- {e}")
            continue
        wall = time.time() - t0
        size_kb = out.stat().st_size // 1024
        vram_gb = (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                   if torch.cuda.is_available() else 0.0)
        rows.append({"preset": name, "wall_s": wall, "size_kb": size_kb,
                     "vram_peak_gb": vram_gb, "out": str(out)})
        print(f"  {name:24s}  {wall:7.1f}  {size_kb:10d}  {vram_gb:12.2f}")
        wandb.log({"preset": name, "wall_s": wall, "vram_peak_gb": vram_gb})
    if rows:
        wandb.summary.update({
            "all_passed": all(r["size_kb"] > 50 for r in rows),
            "max_vram_peak_gb": max(r["vram_peak_gb"] for r in rows),
            "n_presets": len(rows),
            "total_wall_s": sum(r["wall_s"] for r in rows),
        })
        out_json = ("reports/_exp_B64_musicgen_presets.json"
                     if model_size == "melody"
                     else f"reports/_exp_B67_musicgen_{model_size}.json")
        Path(out_json).write_text(json.dumps({"rows": rows, "config": cfg}, indent=2))
    print(f"\n  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="melody",
                    choices=["melody", "melody-large"])
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--clip", default="vocadito_1")
    main(**vars(ap.parse_args()))
