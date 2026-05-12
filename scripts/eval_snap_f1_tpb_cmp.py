"""Snap-F1 cross-check at tpb=12 vs tpb=24 on 9 ASAP pieces.

Snap-F1 was the historical project metric (Phase A-D). Now that Phase E
promoted tpb=12 as the production default, this script verifies that
the snap metric hasn't catastrophically regressed at tpb=12.

For each cached YMT3 prediction:
  1. Apply DP at tpb=24 → q_on24 → score at tpb=24
  2. Apply DP at tpb=12 → q_on12 → score at tpb=24 (after upsample to 12->24=*2)
  3. Apply DP at tpb=12 → q_on12 → score at tpb=12 (matched grid)

The score uses the "snap-allowed quarterLength" metric from
gate_asap_rhythm.py and exp_B12_asap_multi.py.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.beat.octave_sanity import (
    detect_octave_misalignment, apply_octave_correction,
)
from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

CACHE = Path("/workspace/.cache/sweep_e6_features")
ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
PIECES = {
    "asap_Bach__Fugue__bwv_854": "Bach/Fugue/bwv_854",
    "asap_Bach__Fugue__bwv_846": "Bach/Fugue/bwv_846",
    "asap_Bach__Fugue__bwv_848": "Bach/Fugue/bwv_848",
    "asap_Bach__Fugue__bwv_856": "Bach/Fugue/bwv_856",
    "asap_Bach__Fugue__bwv_857": "Bach/Fugue/bwv_857",
    "asap_Beethoven__Piano_Sonatas__21-1": "Beethoven/Piano_Sonatas/21-1",
    "asap_Chopin__Berceuse_op_57": "Chopin/Berceuse_op_57",
    "asap_Liszt__Sonata": "Liszt/Sonata",
    "asap_Schumann__Toccata": "Schumann/Toccata",
}

# allowed snap quarterLength values (from gate_asap_rhythm.py)
ALLOWED_QL = np.array([
    0.0625, 0.083, 0.125, 0.167, 0.1875, 0.25, 0.333,
    0.375, 0.5, 0.667, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
])


def snap_allowed(d: float) -> float:
    """Snap d to the nearest allowed quarterLength."""
    if d <= 0:
        return 0.0625
    return float(ALLOWED_QL[np.argmin(np.abs(ALLOWED_QL - d))])


def _eval_one(piece_key: str, piece_dir: str, *,
              tpb: int, octave_sanity: bool = True,
              eval_seconds: float = 30.0) -> dict | None:
    npz = CACHE / f"{piece_key}.npz"
    gt_path = ASAP_REPO / piece_dir / "midi_score.mid"
    if not (npz.exists() and gt_path.exists()): return None
    d = np.load(npz)
    on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
    beats = d["beats"].copy(); downbeats = d["downbeats"].copy()
    bpm = float(d["bpm"][0]) or 120.0
    notes_ev = [NoteEvent(onset_s=float(on[i]), offset_s=float(off[i]),
                            pitch_midi=int(midi[i]), velocity=80)
                  for i in range(len(on)) if int(midi[i]) >= 1]
    if octave_sanity:
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
                                            tatums_per_beat=tpb,
                                            offgrid_penalty=0.5)
    # Quartile-length per note: (q_off - q_on) / tpb
    pred_ql = (q_off - q_on) / float(tpb)
    pred_ql_snapped = np.array([snap_allowed(d) for d in pred_ql])

    # GT durations (in quarters) from the score MIDI
    pm = pretty_midi.PrettyMIDI(str(gt_path))
    gt_tempo = float(pm.estimate_tempo()) if pm.instruments else 120.0
    sec_per_q = 60.0 / max(gt_tempo, 1e-3)
    gt_durs = []
    for inst in pm.instruments:
        for n in inst.notes:
            if eval_seconds and n.start >= eval_seconds: continue
            gt_durs.append((n.end - n.start) / sec_per_q)
    gt_durs = np.array(gt_durs)
    gt_durs_snapped = np.array([snap_allowed(d) for d in gt_durs])

    # Pair by index-min(): match first min(len(pred), len(gt)) notes
    n = min(len(pred_ql_snapped), len(gt_durs_snapped))
    if n == 0:
        return {"piece": piece_key, "tpb": tpb,
                "snap_pct": 0.0, "n_matched": 0}
    pred_s = pred_ql_snapped[:n]
    gt_s = gt_durs_snapped[:n]
    snap_pct = float(np.mean(pred_s == gt_s))
    return {"piece": piece_key, "tpb": tpb, "snap_pct": snap_pct,
             "n_matched": n}


def main():
    rows = []
    for k, piece_dir in PIECES.items():
        r24 = _eval_one(k, piece_dir, tpb=24, octave_sanity=True)
        r12 = _eval_one(k, piece_dir, tpb=12, octave_sanity=True)
        r8 = _eval_one(k, piece_dir, tpb=8, octave_sanity=True)
        if r24 and r12 and r8:
            print(f"{k:50s} snap@tpb24={r24['snap_pct']:.3f}  "
                  f"snap@tpb12={r12['snap_pct']:.3f}  "
                  f"snap@tpb8={r8['snap_pct']:.3f}  "
                  f"Δ(12-24)={r12['snap_pct']-r24['snap_pct']:+.3f}")
            rows.append({"piece": k,
                          "snap_tpb24": r24['snap_pct'],
                          "snap_tpb12": r12['snap_pct'],
                          "snap_tpb8":  r8['snap_pct']})
    if rows:
        m24 = float(np.mean([r["snap_tpb24"] for r in rows]))
        m12 = float(np.mean([r["snap_tpb12"] for r in rows]))
        m8 = float(np.mean([r["snap_tpb8"] for r in rows]))
        print(f"\nmean snap@tpb24 = {m24:.4f}")
        print(f"mean snap@tpb12 = {m12:.4f}")
        print(f"mean snap@tpb8  = {m8:.4f}")
        print(f"delta tpb24->tpb12 = {m12-m24:+.4f}")
    out = Path("reports/_eval_snap_tpb_cmp.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows,
                                 "mean": {"snap_tpb24": m24 if rows else None,
                                          "snap_tpb12": m12 if rows else None,
                                          "snap_tpb8": m8 if rows else None}},
                                indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
