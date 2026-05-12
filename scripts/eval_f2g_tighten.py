"""Phase F-2g: tighten min_prob to remove per-piece worst-case regression.

F-2e's winning config (min_prob=0.30, search_ms=50) hit -0.053 on voc_8 and
-0.040 on voc_18. F-2g sweeps a narrower grid {min_prob: 0.30..0.50,
search_ms: 25..50} measuring both mean delta AND worst-piece delta. Goal:
mean delta > 0, worst-piece > -0.02 (the v3 spec strict criterion).
"""
from __future__ import annotations
import json, sys
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


def _gt(cid: int):
    csv = VOC / "Annotations" / "Notes" / f"vocadito_{cid}_notesA1.csv"
    out = []
    if not csv.exists(): return out
    for line in csv.read_text().splitlines():
        if not line.strip(): continue
        a, b, c = line.split(",")
        on, freq, dur = float(a), float(b), float(c)
        m = _hz_to_midi(freq)
        if m >= 1: out.append((on, on + dur, m))
    return out


def _f1(pred, gt, tol=0.20):
    if not pred or not gt: return 0.0
    mg, mp = set(), set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in mg: continue
            if abs(po - go) > 0.05: continue
            if pm != gm: continue
            if abs(pf - gf) > tol * (gf - go): continue
            mg.add(j); mp.add(i); break
    tps = len(mp); p = tps / max(len(pred), 1); r = tps / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-6)


def main():
    mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    # Cache prod outputs once
    print("caching production note outputs for 40 clips...")
    cache = {}
    for cid in range(1, 41):
        wav = VOC / "Audio" / f"vocadito_{cid}.wav"
        if not wav.exists(): continue
        y, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        notes_prod = segment_pitch_to_notes(t, hz, vc, mc)
        cache[cid] = {"notes_prod": notes_prod, "y": y, "sr": sr,
                       "gt": _gt(cid)}
        if cid % 10 == 0:
            print(f"  cached {cid}/40")

    print("\ncomputing prod baseline f1s...")
    prod_f1 = {}
    for cid, d in cache.items():
        prod_t = [(n.onset_s, n.offset_s, n.midi()) for n in d["notes_prod"]]
        prod_f1[cid] = _f1(prod_t, d["gt"])
    prod_mean = float(np.mean(list(prod_f1.values())))
    print(f"prod mean off20: {prod_mean:.4f}")

    grid_min_prob = [0.30, 0.35, 0.40, 0.45, 0.50]
    grid_search_ms = [25, 30, 40, 50]
    print(f"\nsweeping {len(grid_min_prob)}x{len(grid_search_ms)} = "
          f"{len(grid_min_prob)*len(grid_search_ms)} cells")
    rows = []
    for mp_th in grid_min_prob:
        for sm in grid_search_ms:
            deltas = []
            for cid, d in cache.items():
                corrected = correct_offsets(d["notes_prod"], d["y"], d["sr"],
                                              min_prob=mp_th, search_ms=sm)
                pred_t = [(n.onset_s, n.offset_s, n.midi()) for n in corrected]
                pf = _f1(pred_t, d["gt"])
                deltas.append({"cid": cid, "delta": pf - prod_f1[cid]})
            mean_d = float(np.mean([r["delta"] for r in deltas]))
            worst = min(deltas, key=lambda r: r["delta"])
            wins = sum(1 for r in deltas if r["delta"] > 0)
            losses = sum(1 for r in deltas if r["delta"] < 0)
            print(f"min_prob={mp_th} search_ms={sm:3d} "
                  f"mean_delta={mean_d:+.4f}  worst=voc_{worst['cid']:2d} "
                  f"({worst['delta']:+.4f})  win/lose={wins}/{losses}")
            rows.append({"min_prob": mp_th, "search_ms": sm,
                          "mean_delta": mean_d,
                          "worst_cid": worst["cid"],
                          "worst_delta": worst["delta"],
                          "wins": wins, "losses": losses})
    out = Path("reports/_phase_f_F2g_tighten.json")
    out.write_text(json.dumps({"prod_mean": prod_mean, "grid": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
