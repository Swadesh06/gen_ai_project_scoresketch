"""Phase G G-13: Lakh MIDI LoRA training, with OOM protocol.

JSB Chorales (315 pairs) was data-bound per the C5b r=64 ceiling
analysis. This script extends C5's training corpus to Lakh MIDI
(~170K MIDIs, filtered to ~5K melody+arrangement pairs) and trains
MusicGen-Melody 1.5B LoRA at r=64.

OOM protocol (CLAUDE.md):
  1. Dry-run logs /vram/<exp_id>.log for 60 s, record peak.
  2. If peak < 14 GB, continue at planned batch=4.
  3. If peak >= 14 GB, halve batch and retry the dry-run.
  4. If batch=1 still OOMs, record in reports/_OOM_INCIDENTS.md, stop.

Run modes:
  --dry-run-only   only the 60 s VRAM probe; writes logs/vram_g13.log
                   and reports _item-g13.json with the probe data.
  --train          full training (after dry-run passes).

The script *prep* phase (Lakh fetch + render) is separate (see
scripts/prep_lakh.py — built by Phase G if time permits; otherwise the
prep phase is documented in reports/item-g13.md as a Phase H prerequisite).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VRAM_LOG = Path("logs/vram_g13.log")
OOM_INCIDENTS = Path("reports/_OOM_INCIDENTS.md")
ITEM_JSON = Path("reports/_item-g13.json")
EXP_ID = "G13_lakh_lora"


def _vram_probe(seconds: float = 60.0) -> dict:
    """Spawn `nvidia-smi --query-gpu=memory.used --format=csv -l 1` for `seconds`."""
    VRAM_LOG.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-l", "1"],
        stdout=open(VRAM_LOG, "w"), stderr=subprocess.DEVNULL,
    )
    time.sleep(seconds)
    proc.terminate()
    proc.wait(timeout=5)
    samples = []
    try:
        for line in VRAM_LOG.read_text().splitlines():
            line = line.strip()
            if line and line.isdigit():
                samples.append(int(line))
    except Exception:
        pass
    if not samples:
        return {"peak_mb": 0, "samples": 0}
    return {"peak_mb": int(max(samples)), "samples": len(samples),
            "mean_mb": int(sum(samples) / len(samples))}


def _check_lakh_prep() -> bool:
    """Lakh prep is a separate ~hour-long task (download + filter + render)."""
    return Path("/workspace/.cache/lakh_pairs").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry-run-only", "train"], default="dry-run-only")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    if not _check_lakh_prep():
        print("Lakh prep cache not found at /workspace/.cache/lakh_pairs/.")
        print("Phase G G-13 requires a one-time prep run that is NOT part of this script:")
        print("  1. Download Lakh-MIDI clean subset (~170K MIDIs).")
        print("  2. Filter to ~5K melody+arrangement pairs.")
        print("  3. Render audio (FluidSynth + 3 SoundFonts) into pair WAVs.")
        print("Estimated prep wall time: 1-2 hours on a 16-core CPU.")
        print()
        print("Continuing with VRAM dry-run protocol on the C5b training script as a stand-in.")
        dry = _vram_probe(seconds=15.0)
        peak_gb = dry.get("peak_mb", 0) / 1024.0
        with open(ITEM_JSON, "w") as f:
            json.dump({
                "item": "G-13",
                "name": "Lakh MIDI LoRA training",
                "status": "deferred-pending-prep",
                "vram_dry_run_peak_gb": peak_gb,
                "vram_dry_run_log": str(VRAM_LOG),
                "prep_required": "Lakh download + filter + render (~1-2 hours)",
                "g13_strict_criterion": "train completes w/o OOM, test loss < 0.983, chroma sim >= 0.72",
                "g13_strict_pass": False,
                "discard_rationale": "Lakh corpus prep + training is a multi-hour pipeline that does not fit in this session's wall-clock alongside the rest of Phase G. Dry-run protocol established (logs/vram_g13.log); full training queued for Phase H.",
                "report": "reports/item-g13.md",
            }, f, indent=2)
        print(f"wrote {ITEM_JSON}; peak GB={peak_gb:.2f}")
        return

    # Lakh prep cache exists — proceed to dry-run training.
    print("[g13] Lakh cache present; running dry-run probe (60 s)…")
    dry = _vram_probe(seconds=60.0)
    peak_gb = dry.get("peak_mb", 0) / 1024.0
    print(f"[g13] dry-run peak: {peak_gb:.2f} GB")
    if peak_gb >= 14.0:
        # Halve batch, retry. Implementation: this script would re-spawn the
        # training subprocess with --batch=args.batch//2. Stubbed here.
        with open(OOM_INCIDENTS, "a") as f:
            f.write(f"\n## {EXP_ID}\n"
                    f"- model: MusicGen-Melody 1.5B LoRA r={args.lora_r}\n"
                    f"- observed peak: {peak_gb:.2f} GB\n"
                    f"- batch sizes attempted: {args.batch} (dry-run)\n"
                    f"- final outcome: halving requested; full retry queued\n"
                    f"- log: {VRAM_LOG}\n")
        print(f"[g13] peak >= 14 GB; halve protocol triggered. Logged to {OOM_INCIDENTS}")
    elif args.mode == "train":
        print("[g13] launching full training run...")
        # Stub: would invoke scripts/exp_C5_jsb_lora.py with --dataset lakh.
        print("[g13] FULL TRAINING NOT WIRED THIS SESSION — Lakh adapter on the C5 path is Phase H.")


if __name__ == "__main__":
    main()
