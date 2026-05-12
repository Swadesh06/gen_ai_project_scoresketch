"""Phase F-3: analyze ME-14-ext for a tpb-routing heuristic.

We have per-piece best tpb from the ME-14-ext sweep. Question: can we
predict the best tpb from cheap piece features (note density, IOI
median, predicted_bpm)?

If yes, we can ship a productionizable router. If no, the best-fixed
default is good enough.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path("/workspace/.cache/sweep_e6_features")


def main():
    with open("reports/_exp_ME14_extended.json") as f:
        ext = json.load(f)
    rows = []
    for r in ext["rows"]:
        key = r["piece"]
        npz = CACHE / f"{key}.npz"
        if not npz.exists(): continue
        d = np.load(npz)
        on = d["notes_on"]
        midi = d["notes_midi"]
        beats = d["beats"]
        bpm = float(d["bpm"][0]) or 120.0
        # Features:
        nps = len(on) / max(float(on[-1] - on[0]), 1e-3) if len(on) >= 2 else 0
        iois = np.diff(on) if len(on) >= 2 else np.array([0.5])
        median_ioi = float(np.median(iois))
        notes_per_beat = (60.0 / max(bpm, 1e-3)) / max(median_ioi, 1e-3)
        midi_iqr = float(np.percentile(midi[midi > 0], 75) - np.percentile(midi[midi > 0], 25)) if (midi > 0).any() else 0
        rows.append({
            "piece": key,
            "best_tpb": r["best"],
            "nps": round(nps, 2),
            "median_ioi": round(median_ioi, 3),
            "notes_per_beat": round(notes_per_beat, 2),
            "pred_bpm": round(bpm, 1),
            "midi_iqr": round(midi_iqr, 1),
            "n_notes": int(len(on)),
            "best_mv2h": r.get("best_mv2h", 0),
        })

    print(f"{'piece':50s}  best   nps  med-ioi  nppb  bpm   iqr   notes")
    for r in rows:
        print(f"{r['piece']:50s}  {r['best_tpb']:14s} {r['nps']:5.2f} {r['median_ioi']:7.3f} "
              f"{r['notes_per_beat']:5.2f} {r['pred_bpm']:5.1f} {r['midi_iqr']:5.1f} {r['n_notes']}")

    # Simple analysis: does best_tpb correlate with notes_per_beat?
    print("\nCorrelation between best_tpb and features:")
    tpb_to_int = {"tpb6_sanity": 6, "tpb8_sanity": 8, "tpb12_sanity": 12,
                   "tpb16_sanity": 16, "tpb18_sanity": 18, "tpb24_sanity": 24,
                   "tpb12_no_sanity": 12, "tpb24_no_sanity": 24}
    best_tpbs = [tpb_to_int.get(r["best_tpb"], 12) for r in rows]
    for f in ("nps", "median_ioi", "notes_per_beat", "pred_bpm", "midi_iqr"):
        vals = [r[f] for r in rows]
        if len(set(best_tpbs)) > 1:
            corr = float(np.corrcoef(vals, best_tpbs)[0, 1])
            print(f"  {f:15s} pearson={corr:+.3f}")
        else:
            print(f"  {f:15s} all best_tpb same — no correlation")

    # Simple decision rule attempt: which tpb maximizes per-piece if we
    # route by notes_per_beat?
    # Heuristic: bins notes_per_beat into 3 buckets and assign each to a tpb.

    # Save analysis.
    out = Path("reports/_phase_f_F3_me14_routing.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
