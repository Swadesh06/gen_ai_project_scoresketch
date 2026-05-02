"""MTG-QBH qualitative gate — runs the humming pipeline on N MTG-QBH clips,
saves SVGs to outputs/mtg_qbh_<mode>/<id>.svg, and logs them to WandB so they
can be eyeballed there. No quantitative threshold; pass = SVGs render and at
least 80% of clips produce >= 1 note."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path

import wandb

from humscribe.config import PipelineConfig
from humscribe.datasets.mtg_qbh import MTGQBH
from humscribe.pipeline import transcribe


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(mtg_dir: str, modes: str, n_clips: int) -> None:
    d = MTGQBH(data_home=mtg_dir)
    tracks = d.load_tracks()
    track_ids = list(tracks.keys())[:n_clips] if n_clips > 0 else list(tracks.keys())
    cfg = {
        "gate": "mtg_qbh_visual",
        "modes": modes,
        "n_clips": len(track_ids),
        "git_sha": git_sha(),
    }
    run = wandb.init(
        project="humscribe-v3.2",
        name=f"gate_mtg_qbh_{modes.replace(',', '_')}_n{len(track_ids)}",
        config=cfg,
        tags=["gate", "mtg_qbh", "visual", "humming"],
        dir="logs/wandb",
    )

    summary_rows: list[dict] = []
    for mode in modes.split(","):
        out_dir = Path(f"outputs/mtg_qbh_{mode}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for tid in track_ids:
            tr = tracks[tid]
            pcfg = PipelineConfig(input_kind="humming", mode=mode)
            r = transcribe(tr.audio_path, pcfg)
            svg_path = out_dir / f"{tid}.svg"
            svg_path.write_text(r.svg)
            row = {
                "mode": mode, "track_id": tid,
                "n_notes": int(r.n_notes), "bpm": float(r.bpm),
                "svg_bytes": len(r.svg),
            }
            summary_rows.append(row)
            wandb.log({f"{mode}/n_notes": r.n_notes, f"{mode}/bpm": r.bpm, "track": tid})
            try:
                wandb.log({f"{mode}/svg/{tid}": wandb.Html(f"<html><body>{r.svg}</body></html>")})
            except Exception:
                pass
            print(f"{tid:8s}/{mode:6s}  notes={r.n_notes:3d}  bpm={r.bpm:6.1f}  svg={svg_path}")

    nonempty = sum(1 for r in summary_rows if r["n_notes"] >= 1)
    pass_pct = nonempty / max(len(summary_rows), 1)
    summary = {
        "total_clips": len(summary_rows),
        "clips_with_notes": nonempty,
        "pct_nonempty": pass_pct,
        "gate_pass": pass_pct >= 0.80,
    }
    wandb.summary.update(summary)
    print(f"\n{nonempty}/{len(summary_rows)} clips produced >= 1 note ({pass_pct*100:.0f}%)")
    print(f"GATE: {'PASS' if summary['gate_pass'] else 'FAIL'}")

    out = Path("reports/_gate_mtg_qbh.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": summary_rows, "config": cfg}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtg-dir", default="~/datasets/mtg_qbh")
    ap.add_argument("--modes", default="soft,medium")
    ap.add_argument("--n-clips", type=int, default=10)
    args = ap.parse_args()
    main(args.mtg_dir, args.modes, args.n_clips)
