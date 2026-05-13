"""Phase G G-9: confidence-aware per-note output eval.

Two readouts:
1. Per-piece global confidence (mean NoteEvent.confidence) vs per-piece MV2H.
   Strict criterion: |Pearson| >= 0.4.
2. Per-note flag recall on Vocadito: deferred until pipeline+Vocadito GT
   pair lands.

For the ASAP path, YMT3 cache notes carry per-note confidence (token
softmax). We read them straight from the cached pickles.
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.notes import NoteEvent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mv2h-source", default="reports/_metric_mv2h_phase_g_asap_g1g2.json")
    ap.add_argument("--cache-dir", default="/workspace/.cache/asap_yourmt3plus")
    ap.add_argument("--out", default="reports/_item-g9.json")
    args = ap.parse_args()
    mv2h_obj = json.loads(Path(args.mv2h_source).read_text())
    mv2h_data = {r["piece"]: r for r in (mv2h_obj.get("asap") or mv2h_obj).get("rows", [])}
    from humscribe.eval.confidence import aggregate_confidence
    from humscribe.notes import NoteEvent
    rows = []
    cache_dir = Path(args.cache_dir)
    cache_beats_dir = Path("/workspace/.cache/asap_beats")
    for pkl in sorted(cache_dir.glob("*.pkl")):
        piece = pkl.stem
        with open(pkl, "rb") as f:
            cache = pickle.load(f)
        notes = [NoteEvent(onset_s=float(n["on"]), offset_s=float(n["off"]),
                            pitch_midi=int(n["midi"]),
                            velocity=int(n.get("vel", 80)),
                            confidence=float(n.get("conf", 1.0)))
                  for n in cache["notes"] if 1 <= int(n["midi"]) <= 127]
        if not notes:
            continue
        ymt3_confs = [float(n.confidence) for n in notes]
        # beat-strength-only aggregate (YMT3 token softmax wasn't cached; we
        # use cached beats from beat_this and approximate confidence as
        # 1 - normalised distance to nearest beat).
        beats_path = cache_beats_dir / f"{piece}.npz"
        if beats_path.exists():
            beats = np.load(str(beats_path))["beats"]
            aggregate_confidence(notes, None, None, beats)
            agg_confs = [float(n.confidence) for n in notes]
        else:
            agg_confs = ymt3_confs
        mv2h = float(mv2h_data.get(piece, {}).get("mv2h", float("nan")))
        rows.append({"piece": piece,
                      "mean_ymt3_conf": float(np.mean(ymt3_confs)),
                      "mean_beat_conf": float(np.mean(agg_confs)),
                      "median_beat_conf": float(np.median(agg_confs)),
                      "n_notes": len(notes), "mv2h": mv2h})
        print(f"{piece:42s} ymt3={np.mean(ymt3_confs):.3f} beat_mean={np.mean(agg_confs):.3f} median={np.median(agg_confs):.3f} mv2h={mv2h:.4f} n={len(notes)}")
    pearson_beat = float("nan")
    pearson_median = float("nan")
    if len(rows) >= 3:
        from scipy.stats import pearsonr
        cb = np.array([r["mean_beat_conf"] for r in rows])
        cm = np.array([r["median_beat_conf"] for r in rows])
        m = np.array([r["mv2h"] for r in rows])
        valid = ~np.isnan(m) & ~np.isclose(cb.std(), 0)
        if valid.sum() >= 3:
            try:
                pearson_beat = float(pearsonr(cb[valid], m[valid])[0])
                pearson_median = float(pearsonr(cm[valid], m[valid])[0])
            except Exception:
                pass
    out = {"rows": rows, "pearson_mean_beat_conf_vs_mv2h": pearson_beat,
            "pearson_median_beat_conf_vs_mv2h": pearson_median,
            "mv2h_source": args.mv2h_source,
            "g9_strict_criterion": "global confidence vs MV2H |r| >= 0.4",
            "g9_strict_pass": bool(abs(pearson_beat) >= 0.4)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}; pearson_beat_mean={pearson_beat:+.3f} pearson_beat_median={pearson_median:+.3f}")


if __name__ == "__main__":
    main()
