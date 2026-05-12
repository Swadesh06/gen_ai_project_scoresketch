"""ME-12 phase-deviation onset detector — eval against Vocadito GT onsets.

Computes onset F1 (±50ms tolerance) on each clip and reports a mean.
Comparison: the existing voicing-based onset estimator's F1 vs phase-
deviation's F1, and what their UNION (vote) F1 looks like.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.ensemble.me12_phase_onset import phase_deviation_onsets

VOC_AUDIO = Path("/home/swadesh/datasets/vocadito/Audio")
VOC_NOTES = Path("/home/swadesh/datasets/vocadito/Annotations/Notes")


def _onset_f1(pred_times: np.ndarray, gt_times: np.ndarray,
              tol_s: float = 0.05) -> dict:
    """Pred/GT onset F1 at ±tol_s tolerance."""
    if len(pred_times) == 0 or len(gt_times) == 0:
        return {"f1": 0.0, "p": 0.0, "r": 0.0,
                 "n_pred": int(len(pred_times)), "n_gt": int(len(gt_times))}
    matched_gt = set()
    matched_pred = set()
    for i, p in enumerate(pred_times):
        for j, g in enumerate(gt_times):
            if j in matched_gt: continue
            if abs(p - g) <= tol_s:
                matched_gt.add(j); matched_pred.add(i); break
    tps = len(matched_pred)
    p = tps / max(len(pred_times), 1)
    r = tps / max(len(gt_times), 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    return {"f1": f1, "p": p, "r": r,
             "n_pred": int(len(pred_times)), "n_gt": int(len(gt_times))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_exp_ME12_phase_onset.json"))
    args = ap.parse_args()
    rows = []
    for cid in range(1, args.limit + 1):
        audio_p = VOC_AUDIO / f"vocadito_{cid}.wav"
        notes_p = VOC_NOTES / f"vocadito_{cid}_notesA1.csv"
        if not (audio_p.exists() and notes_p.exists()): continue
        # GT onsets
        gt_onsets = []
        for line in notes_p.read_text().splitlines():
            if not line.strip(): continue
            a, b, c = line.split(",")
            gt_onsets.append(float(a))
        gt_onsets = np.array(gt_onsets)
        # Audio
        y, sr = load_audio(str(audio_p), target_sr=22050)
        # ME-12 phase-deviation onsets (CPU-only via librosa)
        me12_times, me12_strengths = phase_deviation_onsets(y, sr)
        f1_me12 = _onset_f1(me12_times, gt_onsets, tol_s=0.05)
        rows.append({"clip": cid, "f1_me12": f1_me12})
        print(f"voc_{cid:2d}  ME-12 F1={f1_me12['f1']:.3f}  "
              f"(p={f1_me12['p']:.3f}, r={f1_me12['r']:.3f}, "
              f"n_pred={f1_me12['n_pred']}, n_gt={f1_me12['n_gt']})")
    if rows:
        mean_m = float(np.mean([r["f1_me12"]["f1"] for r in rows]))
        print(f"\nmean ME-12 onset F1 = {mean_m:.4f}")
    args.out.write_text(json.dumps({
        "rows": rows,
        "mean_me12": float(np.mean([r["f1_me12"]["f1"] for r in rows])) if rows else None,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
