"""B40: MTG-QBH visual gate with hybrid voicing. Compare note counts to PESTO."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import wandb

from humscribe.config import PipelineConfig
from humscribe.datasets.mtg_qbh import MTGQBH
from humscribe.pipeline import transcribe


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main(mtg_dir: str, n_clips: int):
    d = MTGQBH(data_home=mtg_dir)
    tracks = d.load_tracks()
    track_ids = list(tracks.keys())[:n_clips]
    cfg = {"exp": "B40_mtg_qbh_hybrid", "git_sha": git_sha(), "n_clips": len(track_ids)}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B40_mtg_qbh_hybrid_n{len(track_ids)}",
                      config=cfg, tags=["B40", "mtg_qbh", "hybrid"], dir="logs/wandb")
    rows = []
    for pm in ("pesto", "pesto_crepevoicing"):
        out_dir = Path(f"outputs/mtg_qbh_hybrid_{pm}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for tid in track_ids:
            tr = tracks[tid]
            pcfg = PipelineConfig(input_kind="humming", mode="soft", pitch_model=pm,
                                   svg_path=str(out_dir / f"{tid}.svg"))
            r = transcribe(tr.audio_path, pcfg)
            rows.append({"track_id": tid, "pitch_model": pm, "n_notes": r.n_notes, "bpm": r.bpm})
            print(f"  {tid:8s}/{pm:22s}  notes={r.n_notes:4d}  bpm={r.bpm:6.1f}")
            wandb.log({f"{pm}/{tid}/n_notes": r.n_notes})
    out = Path("reports/_exp_B40_mtg_qbh_hybrid.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtg-dir", default="~/datasets/mtg_qbh")
    ap.add_argument("--n-clips", type=int, default=10)
    main(**vars(ap.parse_args()))
