"""Phase E session figure: per-piece MV2H improvement from baseline → production.

Renders a horizontal-bar chart showing MV2H for each of the 9 ASAP pieces
under three configurations:
 - DP tpb=24, no octave sanity (session start baseline)
 - DP tpb=24, with octave sanity (after F-1)
 - DP tpb=12, with octave sanity (after ME-14 → production)

Saved to outputs/figures/phase_e_mv2h_progression.png.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Pull cached MV2H values from the eval_octave_sanity_mv2h.py output
    # (tpb=12 + sanity vs tpb=12 no sanity) and from the existing
    # _exp_ME14_mv2h_ensemble.json (tpb=24 no corr).
    with open("reports/_phase_f_F1_octave_mv2h.json") as f:
        oct_data = json.load(f)
    with open("reports/_exp_ME14_mv2h_ensemble.json") as f:
        me14 = json.load(f)

    pieces = []
    base24 = []   # tpb=24 no corr
    sanity24 = []  # tpb=24 + sanity  (= me14's tpb24_sanity_mv2h)
    sanity12 = []  # tpb=12 + sanity (the production config)

    me14_by_piece = {r["piece"]: r for r in me14["rows"]}

    for r in oct_data["rows"]:
        k = r["piece"]
        m14r = me14_by_piece.get(k)
        if not m14r: continue
        base24.append(m14r["tpb24_no_corr_mv2h"] or 0)
        sanity24.append(m14r["tpb24_sanity_mv2h"] or 0)
        sanity12.append(r["mv2h_corr"])
        # Shorten piece name for the chart
        name = k.replace("asap_", "").replace("__", " ").replace("_", " ")
        pieces.append(name)

    # Plot
    y = np.arange(len(pieces))
    height = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))
    bar1 = ax.barh(y - height, base24, height, label="tpb=24, no corrector (baseline)",
                    color="#888888")
    bar2 = ax.barh(y, sanity24, height,
                    label="tpb=24 + octave sanity", color="#5d8bb3")
    bar3 = ax.barh(y + height, sanity12, height,
                    label="tpb=12 + octave sanity (production)", color="#bb4f44")

    ax.set_yticks(y)
    ax.set_yticklabels(pieces, fontsize=9)
    ax.set_xlabel("MV2H (non-aligned DTW)", fontsize=11)
    ax.set_xlim(0, 0.72)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Phase E: ASAP 9-piece MV2H progression (octave-sanity + tpb=12)",
                 fontsize=12)
    ax.axvline(np.mean(base24), color="#888888", linestyle="--", alpha=0.3,
                linewidth=0.5)
    ax.axvline(np.mean(sanity12), color="#bb4f44", linestyle="--", alpha=0.3,
                linewidth=0.5)
    ax.text(0.01, len(pieces) - 0.5, f"mean: baseline {np.mean(base24):.3f} "
                                     f"→ production {np.mean(sanity12):.3f} "
                                     f"(+{np.mean(sanity12) - np.mean(base24):.3f})",
            fontsize=9, color="#444")

    plt.tight_layout()
    out = Path("outputs/figures/phase_e_mv2h_progression.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
