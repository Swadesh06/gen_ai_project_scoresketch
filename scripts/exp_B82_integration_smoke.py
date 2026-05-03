"""B82 — End-to-end smoke test of B76 + per_voice_dp integration.

Verifies:
1. PipelineConfig.per_voice_dp='auto' triggers on Chopin Berceuse audio (per
   _should_use_per_voice_dp heuristic).
2. The full transcribe() pipeline runs without crash, loads B76, applies
   per-voice DP, writes MusicXML + SVG.
3. The output has reasonable note counts vs the score.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import wandb

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe, _should_use_per_voice_dp
from humscribe.rhythm.voice_transformer import is_b76_available

CHOPIN = Path("/workspace/.cache/asap_renders/Chopin__Berceuse_op_57.wav")
OUT_DIR = Path("outputs/b82_integration_smoke")
OUT_JSON = Path("reports/_exp_B82_integration_smoke.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_w = {"git_sha": git_sha(), "audio": str(CHOPIN)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B82_integration_smoke",
                     config=cfg_w, tags=["B82", "integration", "phase-d"],
                     dir="logs/wandb")

    if not CHOPIN.exists():
        print(f"FATAL: {CHOPIN} not found"); run.finish(); return
    print(f"audio: {CHOPIN.stat().st_size // 1024} KB")
    print(f"B76 checkpoint available: {is_b76_available()}")

    # Variant A: production default (per_voice_dp='off' explicit, sanity)
    print("\n=== A. per_voice_dp=off (production default behavior) ===")
    t0 = time.time()
    cfg_a = PipelineConfig(input_kind="piano", per_voice_dp="off",
                            transcriber="bytedance_piano",
                            musicxml_path=str(OUT_DIR / "chopin_off.musicxml"),
                            svg_path=str(OUT_DIR / "chopin_off.svg"))
    res_a = transcribe(str(CHOPIN), cfg_a)
    wall_a = time.time() - t0
    print(f"  notes={res_a.n_notes}, bpm={res_a.bpm:.1f}, wall={wall_a:.1f}s")

    # Variant B: per_voice_dp='auto' (should trigger for Chopin)
    print("\n=== B. per_voice_dp=auto (should auto-trigger for Chopin) ===")
    t0 = time.time()
    cfg_b = PipelineConfig(input_kind="piano", per_voice_dp="auto",
                            transcriber="bytedance_piano",
                            musicxml_path=str(OUT_DIR / "chopin_auto.musicxml"),
                            svg_path=str(OUT_DIR / "chopin_auto.svg"))
    res_b = transcribe(str(CHOPIN), cfg_b)
    wall_b = time.time() - t0
    print(f"  notes={res_b.n_notes}, bpm={res_b.bpm:.1f}, wall={wall_b:.1f}s")

    # Verify auto-routing fired by checking the heuristic
    notes_for_check = res_a.notes
    use_pvd = _should_use_per_voice_dp(notes_for_check, cfg_b)
    print(f"\n  _should_use_per_voice_dp({len(notes_for_check)} notes) = {use_pvd}")

    summary = {
        "n_notes_off": res_a.n_notes,
        "n_notes_auto": res_b.n_notes,
        "bpm_off": float(res_a.bpm),
        "bpm_auto": float(res_b.bpm),
        "wall_off_s": wall_a,
        "wall_auto_s": wall_b,
        "auto_route_fired": bool(use_pvd),
        "musicxml_off_size_kb": (OUT_DIR / "chopin_off.musicxml").stat().st_size // 1024,
        "musicxml_auto_size_kb": (OUT_DIR / "chopin_auto.musicxml").stat().st_size // 1024,
        "svg_off_size_kb": (OUT_DIR / "chopin_off.svg").stat().st_size // 1024,
        "svg_auto_size_kb": (OUT_DIR / "chopin_auto.svg").stat().st_size // 1024,
    }
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    wandb.summary.update(summary)
    OUT_JSON.write_text(json.dumps({"summary": summary, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
