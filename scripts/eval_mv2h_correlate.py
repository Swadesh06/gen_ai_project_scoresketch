"""Phase E item 1 pass criterion: correlation between MV2H and stage-wise metrics.

Reads:
- reports/_metric_mv2h_asap.json (built by eval_mv2h_asap.py)
- reports/_exp_B87b_pipeline_full_asap_tempofix.json (B87b production snap)
- reports/_exp_B12_asap_multi.json (B12 cached snap)

Computes:
- Pearson + Spearman correlation between MV2H and snap on the shared piece set.
- The headline interpretation: does MV2H rank pieces the same way as snap?
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def _pearson(x: list[float], y: list[float]) -> float:
    x = np.array(x); y = np.array(y)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    if len(x) < 2:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mv2h-json", type=Path,
                    default=Path("reports/_metric_mv2h_asap.json"))
    ap.add_argument("--snap-json", type=Path,
                    default=Path("reports/_exp_B87b_pipeline_full_asap_tempofix.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_metric_mv2h_correlate.json"))
    args = ap.parse_args()

    mv = _load(args.mv2h_json)
    sn = _load(args.snap_json)

    # Map by piece key (Bach__Fugue__bwv_846 -> Bach/Fugue/bwv_846).
    def norm(s: str) -> str:
        return s.replace("/", "__")

    sn_by = {norm(r["piece"]): r for r in sn["rows"]}
    mv_by = {r["piece"]: r for r in mv["rows"]}
    shared = sorted(set(mv_by) & set(sn_by))
    if not shared:
        print("no shared pieces — aborting")
        return

    cols = []
    for k in shared:
        snap = float(sn_by[k].get("snap_b87",
                                    sn_by[k].get("snap_b12",
                                                sn_by[k].get("snap", float("nan")))))
        mv2h = float(mv_by[k]["mv2h"])
        cols.append({"piece": k, "snap": snap,
                     "mv2h": mv2h,
                     "mp": float(mv_by[k]["multi_pitch"]),
                     "voice": float(mv_by[k]["voice"]),
                     "meter": float(mv_by[k]["meter"]),
                     "value": float(mv_by[k]["value"]),
                     "harmony": float(mv_by[k]["harmony"])})
    print(f"{'piece':40s}  snap   mv2h    mp    voice  meter  value  harm")
    for c in cols:
        print(f"{c['piece']:40s} {c['snap']:.3f}  {c['mv2h']:.3f}  "
              f"{c['mp']:.3f}  {c['voice']:.3f}  {c['meter']:.3f}  "
              f"{c['value']:.3f}  {c['harmony']:.3f}")

    snap_vals = [c["snap"] for c in cols]
    corr = {}
    for key in ("mv2h", "mp", "voice", "meter", "value", "harmony"):
        vals = [c[key] for c in cols]
        corr[key] = {
            "pearson_vs_snap": _pearson(snap_vals, vals),
            "spearman_vs_snap": _spearman(snap_vals, vals),
        }
    print("\nCorrelation vs snap_b87:")
    for k, v in corr.items():
        print(f"  {k:8s} pearson={v['pearson_vs_snap']:+.3f}  spearman={v['spearman_vs_snap']:+.3f}")

    # Optional: find pieces where MV2H ranks worse than snap suggests (item 1 pass: "at
    # least one example of a MV2H-says-worse piece where note-F1 looked equivalent").
    snap_arr = np.array(snap_vals); mv2h_arr = np.array([c["mv2h"] for c in cols])
    if len(snap_arr) >= 3:
        rank_snap = np.argsort(np.argsort(-snap_arr))
        rank_mv2h = np.argsort(np.argsort(-mv2h_arr))
        delta = rank_mv2h - rank_snap
        i = int(np.argmax(np.abs(delta)))
        anomaly = {"piece": cols[i]["piece"],
                   "snap_rank": int(rank_snap[i]),
                   "mv2h_rank": int(rank_mv2h[i]),
                   "snap": cols[i]["snap"], "mv2h": cols[i]["mv2h"]}
        print(f"\nLargest rank disagreement: {anomaly}")
    else:
        anomaly = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_shared": len(shared), "cols": cols,
        "correlation": corr, "rank_anomaly": anomaly,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
