"""Phase E item 7 ME-4 eval: tonal-meter prior on the DP.

Compares baseline DP (no prior) vs prior-augmented DP scoring on the 5 ASAP
cached pieces. Augmentation: each candidate tatum_pos earns a bonus equal
to `lambda * log P(scale_degree | beat_position_in_bar)` from the Bach
chorale corpus prior.

We don't modify viterbi_quantize_rhythm in-place; instead we re-quantize
using a Python re-implementation that supports a prior callback. If the
prior shows ≥+0.01 MV2H gain on at least 3 pieces, we wire it into the
production DP as a flag.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.ensemble import load_or_build_prior, tonal_meter_log_prior
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

CACHE = Path("/workspace/.cache/sweep_e6_features")
PIECES = [
    "asap_Bach__Fugue__bwv_854",
    "asap_Beethoven__Piano_Sonatas__21-1",
    "asap_Chopin__Berceuse_op_57",
    "asap_Liszt__Sonata",
    "asap_Schumann__Toccata",
]


def _eval_piece(key: str, prior_probs: np.ndarray | None, lambda_p: float,
                tpb: int) -> tuple[float, dict] | None:
    """Return (mv2h, details). prior_probs=None means baseline DP only."""
    npz = CACHE / f"{key}.npz"
    gt = CACHE / f"{key}_gt.txt"
    if not (npz.exists() and gt.exists()):
        return None
    d = np.load(npz)
    on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
    beats = d["beats"]
    bpm = float(d["bpm"][0])
    if bpm <= 0: bpm = 120.0
    if len(on) == 0 or len(beats) < 2:
        return None

    q_on, q_off = viterbi_quantize_rhythm(
        on, off, beats, tatums_per_beat=tpb, offgrid_penalty=0.5,
    )
    # Identify the dominant key for the prior: simple — use the most common
    # pitch class across notes mod 12.
    pcs = (midi % 12)
    pcs_filtered = pcs[(midi > 0)]
    tonic_pc = int(np.bincount(pcs_filtered, minlength=12).argmax())

    if prior_probs is not None and lambda_p > 0:
        # Apply the prior as a post-DP shift: nudge each note's tatum
        # position by ±1 if the alternative position scores higher.
        shifted_q_on = q_on.copy()
        for i in range(len(q_on)):
            m = int(midi[i])
            if m < 1: continue
            cur_pos = int(q_on[i])
            cur_logp = tonal_meter_log_prior(m, cur_pos, tpb, 4, tonic_pc,
                                              prior_probs)
            best_pos, best_logp = cur_pos, cur_logp
            for delta in (-1, 1):
                alt_pos = cur_pos + delta
                if alt_pos < 0: continue
                alt_logp = tonal_meter_log_prior(m, alt_pos, tpb, 4,
                                                  tonic_pc, prior_probs)
                # Trade off DP cost (~1 unit per offset_tatum) vs prior gain.
                # Lambda controls how much we trust the prior vs the DP.
                if lambda_p * (alt_logp - cur_logp) > 1.0:
                    if alt_logp > best_logp:
                        best_pos, best_logp = alt_pos, alt_logp
            shifted_q_on[i] = best_pos
        q_on = shifted_q_on

    tatum_s = 60.0 / (bpm * float(tpb))
    on_origin = float(on[0]) - q_on[0] * tatum_s
    notes = []
    for i in range(len(on)):
        m = int(midi[i])
        if m < 1 or m > 127: continue
        new_on = on_origin + int(q_on[i]) * tatum_s
        new_off = on_origin + int(q_off[i]) * tatum_s
        if new_off <= new_on:
            new_off = new_on + tatum_s
        notes.append(NoteEvent(onset_s=new_on, offset_s=new_off,
                               pitch_midi=m, velocity=80))
    pred = notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4",
                                voices=[0]*len(notes))
    try:
        res = compute_mv2h(pred, gt.read_text(), align="non_aligned",
                            timeout_s=60.0)
    except Exception as e:
        print(f"  {key} mv2h err: {e}"); return None
    return res.mv2h, {"mv2h": res.mv2h, "mp": res.multi_pitch,
                       "voice": res.voice, "meter": res.meter,
                       "value": res.value, "tonic_pc": tonic_pc,
                       "n_notes": len(notes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-prior", type=float, default=2.0)
    ap.add_argument("--tpb", type=int, default=24)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_exp_ME4_tonal_prior.json"))
    args = ap.parse_args()

    prior = load_or_build_prior()
    probs = prior["probs"]
    rows = []
    print(f"=== ME-4 baseline (no prior) vs lambda={args.lambda_prior} ===")
    for k in PIECES:
        t0 = time.time()
        base = _eval_piece(k, None, 0.0, args.tpb)
        m4 = _eval_piece(k, probs, args.lambda_prior, args.tpb)
        if base is None or m4 is None:
            print(f"skip {k}"); continue
        b_mv2h = base[0]; m_mv2h = m4[0]
        delta = m_mv2h - b_mv2h
        row = {"piece": k, "mv2h_base": b_mv2h, "mv2h_me4": m_mv2h,
                "delta": delta, "wall_s": time.time() - t0,
                **{f"base_{k_}": v for k_, v in base[1].items() if k_ != "mv2h"},
                **{f"me4_{k_}": v for k_, v in m4[1].items() if k_ != "mv2h"}}
        rows.append(row)
        print(f"{k:50s} base={b_mv2h:.4f}  ME-4={m_mv2h:.4f}  Δ={delta:+.4f}")

    if rows:
        mean_b = float(np.mean([r["mv2h_base"] for r in rows]))
        mean_m4 = float(np.mean([r["mv2h_me4"] for r in rows]))
        mean_d = mean_m4 - mean_b
        print(f"\nmean baseline MV2H = {mean_b:.4f}")
        print(f"mean ME-4 MV2H     = {mean_m4:.4f}")
        print(f"mean delta         = {mean_d:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "lambda_prior": args.lambda_prior, "tpb": args.tpb,
        "rows": rows,
        "mean_base": float(np.mean([r["mv2h_base"] for r in rows])) if rows else None,
        "mean_me4": float(np.mean([r["mv2h_me4"] for r in rows])) if rows else None,
        "mean_delta": float(np.mean([r["delta"] for r in rows])) if rows else None,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
