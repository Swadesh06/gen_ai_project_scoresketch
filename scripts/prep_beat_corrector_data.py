"""Phase F-1: prep data for the learned beat post-corrector.

Builds (beat_this_output, GT_beat_position) pairs from the 9 ASAP test
pieces. Each beat hypothesis gets a target shift in [-0.2, +0.2] s where
the corrector should move it. Pairs are saved as a single .npz file for
training.

The corrector model trained on this will target the 27pp ASAP score-vs-
real-beats gap. For now we just build the dataset and analyse the
distribution of shifts.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
CACHE = Path("/workspace/.cache/sweep_e6_features")
OUT = Path("/workspace/.cache/beat_corrector_data.npz")

PIECES = {
    "Bach__Fugue__bwv_854": "Bach/Fugue/bwv_854",
    "Bach__Fugue__bwv_846": "Bach/Fugue/bwv_846",
    "Bach__Fugue__bwv_848": "Bach/Fugue/bwv_848",
    "Bach__Fugue__bwv_856": "Bach/Fugue/bwv_856",
    "Bach__Fugue__bwv_857": "Bach/Fugue/bwv_857",
    "Beethoven__Piano_Sonatas__21-1": "Beethoven/Piano_Sonatas/21-1",
    "Chopin__Berceuse_op_57": "Chopin/Berceuse_op_57",
    "Liszt__Sonata": "Liszt/Sonata",
    "Schumann__Toccata": "Schumann/Toccata",
}


def _gt_beats_from_midi(midi_path: Path, eval_seconds: float | None = 30.0) -> np.ndarray:
    """Extract GT beat times by reading the time signature and tempo."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    end_time = pm.get_end_time()
    if eval_seconds:
        end_time = min(end_time, eval_seconds)
    beats = pm.get_beats(start_time=0.0)
    return beats[beats < end_time]


def _match_beats(pred_beats: np.ndarray, gt_beats: np.ndarray,
                 tol_s: float = 0.5) -> list[tuple[float, float]]:
    """For each pred beat, find nearest GT beat within tol. Returns
    (pred_time, gt_time) pairs."""
    pairs = []
    used = set()
    for p in pred_beats:
        if len(gt_beats) == 0:
            continue
        idx = int(np.argmin(np.abs(gt_beats - p)))
        if idx in used:
            continue
        d = float(gt_beats[idx] - p)
        if abs(d) > tol_s:
            continue
        pairs.append((float(p), float(gt_beats[idx])))
        used.add(idx)
    return pairs


def main():
    all_rows = []
    for key, piece_dir in PIECES.items():
        gt_path = ASAP_REPO / piece_dir / "midi_score.mid"
        npz = CACHE / f"asap_{key}.npz"
        if not gt_path.exists() or not npz.exists():
            print(f"skip {key}: missing data"); continue
        d = np.load(npz)
        pred_beats = d["beats"]
        gt_beats = _gt_beats_from_midi(gt_path, eval_seconds=30.0)
        pairs = _match_beats(pred_beats, gt_beats, tol_s=0.5)
        shifts = [g - p for p, g in pairs]
        if not pairs:
            print(f"skip {key}: no matched beats")
            continue
        print(f"{key:42s} n_pred={len(pred_beats):3d} n_gt={len(gt_beats):3d} "
              f"matched={len(pairs):3d} mean_shift={np.mean(shifts):+.3f}s "
              f"std_shift={np.std(shifts):.3f}s "
              f"max|shift|={np.max(np.abs(shifts)):.3f}s")
        for (p, g) in pairs:
            all_rows.append({"piece": key, "pred_beat": p, "gt_beat": g,
                              "shift": g - p})

    # Save dataset.
    if all_rows:
        np.savez(str(OUT),
                  pred_beats=np.array([r["pred_beat"] for r in all_rows], dtype=np.float64),
                  gt_beats=np.array([r["gt_beat"] for r in all_rows], dtype=np.float64),
                  shifts=np.array([r["shift"] for r in all_rows], dtype=np.float64),
                  pieces=np.array([r["piece"] for r in all_rows]))
        shifts = np.array([r["shift"] for r in all_rows])
        print(f"\nDataset: {len(all_rows)} (pred, gt) pairs")
        print(f"  mean shift   = {shifts.mean():+.4f} s")
        print(f"  std shift    = {shifts.std():.4f} s")
        print(f"  median |shift| = {float(np.median(np.abs(shifts))):.4f} s")
        print(f"  max |shift|  = {float(np.max(np.abs(shifts))):.4f} s")
        print(f"  shifts in ±0.05 s: {int((np.abs(shifts) < 0.05).sum())} ({100*(np.abs(shifts) < 0.05).mean():.1f}%)")
        print(f"  shifts in ±0.20 s: {int((np.abs(shifts) < 0.20).sum())} ({100*(np.abs(shifts) < 0.20).mean():.1f}%)")
        print(f"saved to {OUT}")
    else:
        print("no data")


if __name__ == "__main__":
    main()
