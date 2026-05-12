"""Local random sweep over the MV2H-e6 parameter space, 200 trials.

Extends the WandB Bayesian sweep that hit +0.022 vs baseline without
clearing the v3 +0.03 criterion. Uses cached features under
/workspace/.cache/sweep_e6_features/ — pure CPU, ~30 s per trial.

Param ranges widened vs the original yaml:
- complexity_alpha: 0.2 .. 4.0  (was 0.5 .. 3.0)
- sigma_quant:     0.01 .. 0.10 (was 0.02 .. 0.06)
- dp_offgrid:      0.1 .. 3.0   (was 0.25 .. 1.5)
- voicing_vt:      0.60 .. 0.90 (was 0.65 .. 0.85)

Writes reports/_phase_e_item6_local_sweep.json with full trial log.
"""
from __future__ import annotations
import json, random, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.config import ModeConfig
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm


CACHE = Path("/workspace/.cache/sweep_e6_features")
OUT = Path("reports/_phase_e_item6_local_sweep.json")

ASAP_KEYS = [
    "Bach__Fugue__bwv_854",
    "Beethoven__Piano_Sonatas__21-1",
    "Chopin__Berceuse_op_57",
    "Liszt__Sonata",
    "Schumann__Toccata",
]
VOC_IDS = list(range(1, 11))


def _eval_asap(key, tpb, sigma_quant, target_bpm, dp_off):
    npz = CACHE / f"asap_{key}.npz"
    gt = CACHE / f"asap_{key}_gt.txt"
    if not (npz.exists() and gt.exists()): return None
    d = np.load(npz)
    on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
    beats = d["beats"]
    if len(beats) < 2 or len(on) == 0: return None
    q_on, q_off = viterbi_quantize_rhythm(
        on, off, beats, tatums_per_beat=int(tpb),
        offgrid_penalty=float(dp_off),
    )
    bpm = float(d["bpm"][0]) or 120.0
    tatum_s = 60.0 / (bpm * float(tpb))
    on_origin = float(on[0]) - q_on[0] * tatum_s if len(on) > 0 else 0.0
    notes = []
    for i in range(len(on)):
        m = int(midi[i])
        if m < 1 or m > 127: continue
        new_on = on_origin + int(q_on[i]) * tatum_s
        new_off = on_origin + int(q_off[i]) * tatum_s
        if new_off <= new_on: new_off = new_on + tatum_s
        notes.append(NoteEvent(onset_s=new_on, offset_s=new_off, pitch_midi=m, velocity=80))
    pred_text = notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4", voices=[0]*len(notes))
    try:
        res = compute_mv2h(pred_text, gt.read_text(), align="non_aligned", timeout_s=60.0)
        return res.mv2h
    except Exception:
        return None


def _eval_voc(vid, tpb, sigma_quant, target_bpm, voicing_psw, voicing_vt, dp_off):
    npz = CACHE / f"voc_{vid}.npz"
    gt = CACHE / f"voc_{vid}_A1_gt.txt"
    if not (npz.exists() and gt.exists()): return None
    d = np.load(npz)
    t = d["t"]; hz = d["hz"]; vc = d["vc"]
    beats = d["beats"]
    mc = ModeConfig(voicing_threshold=float(voicing_vt),
                    min_note_seconds=0.052,
                    onset_merge_seconds=0.026,
                    dp_offgrid_penalty=float(dp_off),
                    pitch_smooth_window=int(voicing_psw))
    notes = segment_pitch_to_notes(t, hz, vc, mc)
    if not notes: return None
    bpm = float(d["bpm"][0]) or 120.0
    on = np.array([n.onset_s for n in notes])
    off = np.array([n.offset_s for n in notes])
    if len(beats) >= 2 and len(on) > 0:
        q_on, q_off = viterbi_quantize_rhythm(
            on, off, beats, tatums_per_beat=int(tpb),
            offgrid_penalty=float(dp_off),
        )
        tatum_s = 60.0 / (bpm * float(tpb))
        on_origin = float(on[0]) - q_on[0] * tatum_s
        new_notes = []
        for i, n in enumerate(notes):
            new_on = on_origin + int(q_on[i]) * tatum_s
            new_off = on_origin + int(q_off[i]) * tatum_s
            if new_off <= new_on: new_off = new_on + tatum_s
            new_notes.append(NoteEvent(onset_s=new_on, offset_s=new_off,
                                       pitch_midi=n.midi(), velocity=n.velocity))
        notes = new_notes
    pred_text = notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4", voices=[0]*len(notes))
    try:
        res = compute_mv2h(pred_text, gt.read_text(), align="non_aligned", timeout_s=60.0)
        return res.mv2h
    except Exception:
        return None


def main():
    n_trials = 200
    random.seed(42)
    trials = []
    t_start = time.time()
    best_overall = -1.0
    for i in range(n_trials):
        cfg = {
            "tpb": random.choice([6, 8, 12, 16, 24]),
            "complexity_alpha": random.uniform(0.2, 4.0),
            "sigma_quant": random.uniform(0.01, 0.10),
            "voicing_psw": random.choice([11, 13, 15, 17, 19, 21, 23]),
            "voicing_vt": random.uniform(0.60, 0.90),
            "target_bpm": random.uniform(80, 130),
            "dp_off": random.uniform(0.1, 3.0),
        }
        asap_scores = []
        for k in ASAP_KEYS:
            r = _eval_asap(k, cfg["tpb"], cfg["sigma_quant"],
                           cfg["target_bpm"], cfg["dp_off"])
            if r is not None: asap_scores.append(r)
        voc_scores = []
        for v in VOC_IDS:
            r = _eval_voc(v, cfg["tpb"], cfg["sigma_quant"],
                          cfg["target_bpm"], cfg["voicing_psw"],
                          cfg["voicing_vt"], cfg["dp_off"])
            if r is not None: voc_scores.append(r)
        if not asap_scores or not voc_scores: continue
        asap_mean = float(np.mean(asap_scores))
        voc_mean = float(np.mean(voc_scores))
        overall = float(np.mean(asap_scores + voc_scores))
        trials.append({"cfg": cfg, "asap_mean": asap_mean, "voc_mean": voc_mean,
                       "overall": overall, "trial": i})
        if overall > best_overall:
            best_overall = overall
            print(f"trial {i:3d}: overall={overall:.4f}  asap={asap_mean:.4f}  "
                  f"voc={voc_mean:.4f}  tpb={cfg['tpb']} alpha={cfg['complexity_alpha']:.2f} "
                  f"vt={cfg['voicing_vt']:.3f} psw={cfg['voicing_psw']} "
                  f"target={cfg['target_bpm']:.1f} dp_off={cfg['dp_off']:.2f}  *NEW BEST*")
        elif i % 25 == 0:
            print(f"trial {i:3d}: overall={overall:.4f}  (best so far={best_overall:.4f})")
        if (i + 1) % 50 == 0:
            OUT.write_text(json.dumps({"trials": trials,
                                       "best_overall": best_overall},
                                      indent=2))
    OUT.write_text(json.dumps({"trials": trials,
                               "best_overall": best_overall}, indent=2))
    wall = time.time() - t_start
    print(f"\nwall: {wall:.1f}s ({wall/60:.1f} min)  best overall: {best_overall:.4f}")
    print(f"baseline reference: 0.5074 (per item-6 report)")
    print(f"v3 +0.03 target: 0.5374")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
