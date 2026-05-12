"""Verify the F-2e production wiring on the full 40-clip Vocadito set."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.formant_corrector import correct_offsets
from humscribe.pitch.voicing import segment_pitch_to_notes

VOC = Path("/home/swadesh/datasets/vocadito")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0: return -1
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _f1(pred, gt, tol_off_rel=0.20):
    if not pred or not gt: return 0.0
    mg, mp = set(), set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in mg: continue
            if abs(po - go) > 0.05: continue
            if pm != gm: continue
            if abs(pf - gf) > tol_off_rel * (gf - go): continue
            mg.add(j); mp.add(i); break
    tps = len(mp); p = tps / max(len(pred), 1); r = tps / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-6)


def main():
    mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    results = []
    for cid in range(1, 41):
        wav = VOC / "Audio" / f"vocadito_{cid}.wav"
        csv = VOC / "Annotations" / "Notes" / f"vocadito_{cid}_notesA1.csv"
        if not (wav.exists() and csv.exists()): continue
        gt = []
        for line in csv.read_text().splitlines():
            if not line.strip(): continue
            a, b, c = line.split(",")
            on, freq, dur = float(a), float(b), float(c)
            m = _hz_to_midi(freq)
            if m >= 1: gt.append((on, on + dur, m))
        y, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        notes_prod = segment_pitch_to_notes(t, hz, vc, mc)
        notes_f2e = correct_offsets(notes_prod, y, sr,
                                     min_prob=0.30, search_ms=50.0)
        prod_t = [(n.onset_s, n.offset_s, n.midi()) for n in notes_prod]
        f2e_t = [(n.onset_s, n.offset_s, n.midi()) for n in notes_f2e]
        pp = _f1(prod_t, gt); pf = _f1(f2e_t, gt)
        results.append({"cid": cid, "prod": pp, "f2e": pf, "delta": pf - pp})
        print(f"voc_{cid:2d}: prod={pp:.3f}  f2e={pf:.3f}  Δ={pf-pp:+.3f}")
    mp = float(np.mean([r["prod"] for r in results]))
    mf = float(np.mean([r["f2e"] for r in results]))
    print(f"\nmean prod off20: {mp:.4f}")
    print(f"mean f2e off20:  {mf:.4f}")
    print(f"delta:           {mf - mp:+.4f}")
    wins = sum(1 for r in results if r["delta"] > 0)
    losses = sum(1 for r in results if r["delta"] < 0)
    sames = len(results) - wins - losses
    print(f"win/lose/same: {wins}/{losses}/{sames}")
    losses_sorted = sorted(
        (r for r in results if r["delta"] < 0), key=lambda r: r["delta"]
    )
    print("\nworst regressions (sorted by delta):")
    for r in losses_sorted[:10]:
        print(f"  voc_{r['cid']:2d}: prod={r['prod']:.3f} → f2e={r['f2e']:.3f}  Δ={r['delta']:+.3f}")
    import json
    out = Path("reports/_phase_f_F2e_production_verify.json")
    out.write_text(json.dumps({
        "config": {"min_prob": 0.30, "search_ms": 50.0},
        "mean_prod_off20": mp, "mean_f2e_off20": mf, "delta": mf - mp,
        "wins": wins, "losses": losses, "sames": sames,
        "per_piece": results,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
