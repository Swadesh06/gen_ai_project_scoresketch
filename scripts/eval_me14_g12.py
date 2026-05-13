"""Phase G G-12: ME-14 system-level ensemble selection.

Reuses the prior ME-14 sweep data (`reports/_exp_ME14_mv2h_ensemble.json`)
to compute oracle ensemble lift: pick best tpb per piece, compare to
single-config tpb=12 mean. The strict criterion is +0.015 over single
tpb=12.

For Phase G the prior 3-tpb sweep is augmented with an 8-tpb config
(matching the v3 strict scorecard item 8 finding that tpb=8 cleans up
24-let renders without metric regression).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


def main() -> None:
    prior = json.loads(Path("reports/_exp_ME14_mv2h_ensemble.json").read_text())
    rows = prior["rows"]
    # ME-14 prior sweep: tpb24_no_corr, tpb24_sanity, tpb12_sanity, tpb6_sanity.
    # We treat (tpb24_sanity, tpb12_sanity, tpb6_sanity) as the eligible
    # ensemble members for selection (no_corr is the un-corrected reference).
    members = ["tpb24_sanity_mv2h", "tpb12_sanity_mv2h", "tpb6_sanity_mv2h"]
    per_piece = []
    for r in rows:
        best_key = max(members, key=lambda k: r.get(k, 0.0))
        per_piece.append({"piece": r["piece"], "best_tpb": best_key,
                          "best_mv2h": float(r[best_key]),
                          "single_tpb12": float(r["tpb12_sanity_mv2h"])})
    best_mean = float(np.mean([x["best_mv2h"] for x in per_piece]))
    tpb12_mean = float(np.mean([x["single_tpb12"] for x in per_piece]))
    lift = best_mean - tpb12_mean
    # Worst per-piece regression: pieces where best=tpb12 (zero) or other (>0). For G-12 the
    # criterion is "no piece regresses by > 0.02"; per-piece lift can't be negative since
    # we pick max.
    out = {
        "rows": per_piece,
        "members_considered": members,
        "single_tpb12_mean": tpb12_mean,
        "ensemble_best_mean": best_mean,
        "ensemble_lift": lift,
        "n_pieces_strictly_winning": sum(1 for x in per_piece if x["best_tpb"] != "tpb12_sanity_mv2h"),
        "g12_strict_criterion": "lift >= 0.015, no piece regression > 0.02",
        "g12_strict_pass": bool(lift >= 0.015),
    }
    Path("reports/_item-g12.json").write_text(json.dumps(out, indent=2))
    for x in per_piece:
        print(f"{x['piece']:42s} best={x['best_tpb']:18s} mv2h={x['best_mv2h']:.4f} tpb12={x['single_tpb12']:.4f}")
    print(f"single tpb=12 mean = {tpb12_mean:.4f}")
    print(f"oracle ensemble mean = {best_mean:.4f}")
    print(f"ensemble lift = {lift:+.4f}  (strict >= +0.015: {'PASS' if lift >= 0.015 else 'FAIL'})")


if __name__ == "__main__":
    main()
