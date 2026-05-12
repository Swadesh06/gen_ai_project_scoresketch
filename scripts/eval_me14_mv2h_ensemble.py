"""Phase E item 7 ME-14 + Phase F-3: MV2H-driven system-level ensemble.

Run the pipeline at N configurations on each test piece, score each
output via MV2H, pick the best per piece, and aggregate. The 'best
config per piece' decision can then be routed by piece-feature signals
in production.

For now we test 4 configs on the 9 ASAP cached pieces:
- baseline (tpb=24, no DP correction beyond default)
- octave_sanity on (default; expected best per F-1 result)
- tpb=12 (lower resolution, simpler tuplets)
- tpb=6  (coarsest)

Goal: confirm octave_sanity is the right default and quantify the
per-piece win rate.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.beat.octave_sanity import (
    detect_octave_misalignment, apply_octave_correction,
)
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

CACHE = Path("/workspace/.cache/sweep_e6_features")
PIECES = [
    "asap_Bach__Fugue__bwv_854",
    "asap_Bach__Fugue__bwv_846",
    "asap_Bach__Fugue__bwv_848",
    "asap_Bach__Fugue__bwv_856",
    "asap_Bach__Fugue__bwv_857",
    "asap_Beethoven__Piano_Sonatas__21-1",
    "asap_Chopin__Berceuse_op_57",
    "asap_Liszt__Sonata",
    "asap_Schumann__Toccata",
]

CONFIGS = [
    {"name": "tpb24_no_corr", "tpb": 24, "octave_sanity": False},
    {"name": "tpb24_sanity",  "tpb": 24, "octave_sanity": True},
    {"name": "tpb12_sanity",  "tpb": 12, "octave_sanity": True},
    {"name": "tpb6_sanity",   "tpb": 6,  "octave_sanity": True},
]


def _run_config(piece_key: str, cfg: dict) -> float | None:
    npz = CACHE / f"{piece_key}.npz"
    gt = CACHE / f"{piece_key}_gt.txt"
    if not (npz.exists() and gt.exists()): return None
    d = np.load(npz)
    on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
    beats = d["beats"].copy(); downbeats = d["downbeats"].copy()
    bpm = float(d["bpm"][0]) or 120.0
    notes_ev = []
    for i in range(len(on)):
        m = int(midi[i])
        if m < 1: continue
        notes_ev.append(NoteEvent(onset_s=float(on[i]),
                                    offset_s=float(off[i]),
                                    pitch_midi=m, velocity=80))
    if cfg["octave_sanity"]:
        diag = detect_octave_misalignment(beats, notes_ev)
        beats, downbeats = apply_octave_correction(beats, downbeats,
                                                     diag["recommend"])
        if len(beats) >= 2:
            ibis = np.diff(beats)
            ibis = ibis[(ibis > 0.01) & (ibis < 5.0)]
            if len(ibis) > 0:
                bpm = 60.0 / float(np.median(ibis))
    if len(beats) < 2 or len(on) == 0: return None
    q_on, q_off = viterbi_quantize_rhythm(on, off, beats,
                                            tatums_per_beat=cfg["tpb"],
                                            offgrid_penalty=0.5)
    tatum_s = 60.0 / (bpm * cfg["tpb"])
    on_origin = float(on[0]) - q_on[0] * tatum_s
    notes_out = []
    for i in range(len(on)):
        m = int(midi[i])
        if m < 1: continue
        new_on = on_origin + int(q_on[i]) * tatum_s
        new_off = on_origin + int(q_off[i]) * tatum_s
        if new_off <= new_on: new_off = new_on + tatum_s
        notes_out.append(NoteEvent(onset_s=new_on, offset_s=new_off,
                                    pitch_midi=m, velocity=80))
    pred = notes_to_mv2h_format(notes_out, bpm=bpm, time_sig="4/4",
                                voices=[0]*len(notes_out))
    try:
        return compute_mv2h(pred, gt.read_text(), align="non_aligned",
                              timeout_s=60.0).mv2h
    except Exception as e:
        return None


def main():
    rows = []
    print(f"{'piece':50s}  " + "  ".join(f"{c['name']:>16}" for c in CONFIGS) + "  best")
    for k in PIECES:
        scores = {}
        for cfg in CONFIGS:
            scores[cfg["name"]] = _run_config(k, cfg)
        best_name = max(scores, key=lambda n: scores[n] if scores[n] is not None else -1)
        best_score = scores[best_name]
        rows.append({"piece": k, **{f"{n}_mv2h": s for n, s in scores.items()},
                      "best": best_name, "best_mv2h": best_score})
        cells = "  ".join(f"{(scores[c['name']] or 0):16.4f}" for c in CONFIGS)
        print(f"{k:50s}  {cells}  {best_name}")

    # Aggregate
    means = {c["name"]: float(np.nanmean([s for r in rows
                                             if (s:=r[f'{c["name"]}_mv2h']) is not None]))
              for c in CONFIGS}
    print("\nMean MV2H per config:")
    for n, v in sorted(means.items(), key=lambda x: -x[1]):
        print(f"  {n:20s}  {v:.4f}")

    # Ensemble: pick best per piece, what's the mean?
    ens = float(np.mean([r["best_mv2h"] for r in rows if r["best_mv2h"] is not None]))
    print(f"\nOracle-best per piece ensemble: {ens:.4f}")
    out = Path("reports/_exp_ME14_mv2h_ensemble.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "means": means,
                                "oracle_ensemble_mv2h": ens}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
