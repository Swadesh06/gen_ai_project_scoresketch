"""Phase F-1: validate the octave-sanity corrector by measuring MV2H delta.

For each of the 9 ASAP pieces, apply the corrector to beat_this's output
and compare MV2H of the corrected DP output vs the uncorrected baseline.
Headline: does the corrector close any of the 27pp score-vs-real-beats
gap?
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


def _eval(piece_key: str, apply_corr: bool, tpb: int = 24) -> dict | None:
    npz = CACHE / f"{piece_key}.npz"
    gt = CACHE / f"{piece_key}_gt.txt"
    if not (npz.exists() and gt.exists()): return None
    d = np.load(npz)
    on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
    beats = d["beats"].copy(); downbeats = d["downbeats"].copy()
    bpm = float(d["bpm"][0]);
    if bpm <= 0: bpm = 120.0
    notes_ev = []
    for i in range(len(on)):
        m = int(midi[i])
        if m < 1: continue
        notes_ev.append(NoteEvent(onset_s=float(on[i]),
                                    offset_s=float(off[i]),
                                    pitch_midi=m, velocity=80))
    rec = "keep"
    if apply_corr:
        diag = detect_octave_misalignment(beats, notes_ev)
        rec = diag["recommend"]
        beats, downbeats = apply_octave_correction(beats, downbeats, rec)
        # Recompute bpm after correction.
        if len(beats) >= 2:
            ibis = np.diff(beats)
            ibis = ibis[(ibis > 0.01) & (ibis < 5.0)]
            if len(ibis) > 0:
                bpm = 60.0 / float(np.median(ibis))
    # Apply DP with these (possibly corrected) beats.
    if len(beats) < 2 or len(on) == 0:
        return None
    q_on, q_off = viterbi_quantize_rhythm(on, off, beats,
                                            tatums_per_beat=tpb,
                                            offgrid_penalty=0.5)
    tatum_s = 60.0 / (bpm * tpb)
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
        res = compute_mv2h(pred, gt.read_text(), align="non_aligned", timeout_s=60.0)
    except Exception as e:
        return {"piece": piece_key, "recommend": rec, "error": str(e)}
    return {"piece": piece_key, "recommend": rec, "bpm": bpm,
             **res.as_dict()}


def main():
    rows = []
    for k in PIECES:
        base = _eval(k, apply_corr=False)
        corr = _eval(k, apply_corr=True)
        if base is None or corr is None: continue
        delta = corr.get("mv2h", 0) - base.get("mv2h", 0)
        rows.append({"piece": k, "recommend": corr.get("recommend"),
                      "mv2h_base": base.get("mv2h"),
                      "mv2h_corr": corr.get("mv2h"),
                      "delta": delta,
                      "bpm_base": base.get("bpm"),
                      "bpm_corr": corr.get("bpm")})
        print(f"{k:50s} rec={corr.get('recommend'):6s}  "
              f"mv2h: {base['mv2h']:.4f} → {corr['mv2h']:.4f}  "
              f"Δ={delta:+.4f}  bpm: {base['bpm']:.0f} → {corr['bpm']:.0f}")
    if rows:
        b = float(np.mean([r["mv2h_base"] for r in rows]))
        c = float(np.mean([r["mv2h_corr"] for r in rows]))
        print(f"\nmean baseline MV2H = {b:.4f}")
        print(f"mean corrected MV2H = {c:.4f}")
        print(f"mean delta = {c-b:+.4f}")
    out = Path("reports/_phase_f_F1_octave_mv2h.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows,
                                "mean_base": float(np.mean([r["mv2h_base"] for r in rows])) if rows else None,
                                "mean_corr": float(np.mean([r["mv2h_corr"] for r in rows])) if rows else None,
                                "mean_delta": float(np.mean([r["delta"] for r in rows])) if rows else None}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
