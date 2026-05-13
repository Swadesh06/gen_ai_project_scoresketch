"""Phase G G-10: bar-level structural consistency diagnostic.

For each ASAP piece, run beat_this on cached audio (use cache), compute
bar_consistency on the downbeats, and correlate with MV2H. Pass criterion:
score < 0.4 on Liszt Sonata (catches structural inconsistency),
> 0.8 on Bach Fugues, correlation with MV2H ≥ 0.3.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.eval.bar_diag import bar_consistency, beat_consistency

CACHE_BEATS = Path("/workspace/.cache/asap_beats")
CACHE_RENDERS = Path("/workspace/.cache/asap_renders")
PIECES = [
    "Bach__Fugue__bwv_846", "Bach__Fugue__bwv_848", "Bach__Fugue__bwv_854",
    "Bach__Fugue__bwv_856", "Bach__Fugue__bwv_857",
    "Beethoven__Piano_Sonatas__21-1", "Schumann__Toccata",
    "Chopin__Berceuse_op_57", "Liszt__Sonata",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/_item-g10.json")
    ap.add_argument("--mv2h-source", default="reports/_metric_mv2h_phase_g_asap_g1g2.json")
    args = ap.parse_args()
    mv2h_data = {}
    if Path(args.mv2h_source).exists():
        mv2h_obj = json.loads(Path(args.mv2h_source).read_text())
        for r in (mv2h_obj.get("asap") or mv2h_obj).get("rows", []):
            mv2h_data[r["piece"]] = r
    rows = []
    for piece in PIECES:
        npz = CACHE_BEATS / f"{piece}.npz"
        if not npz.exists():
            from humscribe.beat.beat_this_track import track_beats_beat_this
            wav = CACHE_RENDERS / f"{piece}.wav"
            beats, downbeats, bpm = track_beats_beat_this(str(wav), target_bpm=110.0)
            np.savez(str(npz), beats=beats, downbeats=downbeats, bpm=np.float64(bpm))
        d = np.load(str(npz))
        bar_c = bar_consistency(d["beats"], d["downbeats"])
        beat_c = beat_consistency(d["beats"])
        mv2h = mv2h_data.get(piece, {}).get("mv2h", float("nan"))
        rows.append({"piece": piece, "bar_consistency": bar_c,
                      "beat_consistency": beat_c, "mv2h": mv2h,
                      "n_beats": int(len(d["beats"])),
                      "n_downbeats": int(len(d["downbeats"]))})
        print(f"{piece:42s} bar_c={bar_c:.3f} beat_c={beat_c:.3f} mv2h={mv2h:.4f}")
    pearson_bar = float("nan")
    pearson_beat = float("nan")
    if len(rows) >= 3:
        b = np.array([r["bar_consistency"] for r in rows])
        bt = np.array([r["beat_consistency"] for r in rows])
        m = np.array([r["mv2h"] for r in rows])
        valid = ~np.isnan(m)
        if valid.sum() >= 3:
            from scipy.stats import pearsonr
            pearson_bar = float(pearsonr(b[valid], m[valid])[0])
            pearson_beat = float(pearsonr(bt[valid], m[valid])[0])
    out = {"rows": rows, "pearson_bar_vs_mv2h": pearson_bar,
            "pearson_beat_vs_mv2h": pearson_beat,
            "mv2h_source": args.mv2h_source}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}; pearson(bar, mv2h)={pearson_bar:+.3f}, pearson(beat, mv2h)={pearson_beat:+.3f}")


if __name__ == "__main__":
    main()
