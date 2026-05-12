"""Phase F-2e threshold sweep: find a winning min_prob / search_ms combo.

F-2e default config (min_prob=0.5, search_ms=100) was Δ = -0.010. Sweep
min_prob and search_ms to find a combo where Δ > +0.01.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.train.formant_offset import FormantOffsetBiLSTM, FormantOffsetConfig
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes

VOC_AUDIO = Path("/home/swadesh/datasets/vocadito/Audio")
VOC_NOTES = Path("/home/swadesh/datasets/vocadito/Annotations/Notes")
FORMANT_CACHE = Path("/workspace/.cache/voc_formant")
CKPT_DIR = Path("checkpoints/formant_offset_vocadito")
OUT = Path("reports/_phase_f_F2e_threshold_sweep.json")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0: return -1
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _gt_notes(clip_id: int):
    csv = VOC_NOTES / f"vocadito_{clip_id}_notesA1.csv"
    out = []
    if not csv.exists(): return out
    for line in csv.read_text().splitlines():
        if not line.strip(): continue
        a, b, c = line.split(",")
        on, freq, dur = float(a), float(b), float(c)
        m = _hz_to_midi(freq)
        if m >= 1: out.append((on, on + dur, m))
    return out


def _probs(model, mel):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(mel.T).unsqueeze(0)).squeeze(0).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def _snap(heur_off, probs, search_ms, min_prob, hop_s=0.01):
    n = len(probs)
    center = int(heur_off / hop_s)
    window = int(search_ms / 1000.0 / hop_s)
    lo, hi = max(0, center - window), min(n, center + window + 1)
    if lo >= hi: return heur_off
    sub = probs[lo:hi]
    idx = int(np.argmax(sub))
    if sub[idx] < min_prob: return heur_off
    return (lo + idx) * hop_s


def _f1(pred, gt, offset_rel_tol=0.20):
    if not pred or not gt: return 0.0
    mg, mp = set(), set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in mg: continue
            if abs(po - go) > 0.05: continue
            if pm != gm: continue
            if abs(pf - gf) > offset_rel_tol * (gf - go): continue
            mg.add(j); mp.add(i); break
    tps = len(mp)
    p = tps / max(len(pred), 1); r = tps / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-6)


def main():
    cfg = FormantOffsetConfig(in_dim=80, hidden=96, layers=2, dropout=0.2)
    folds, val_ids = {}, {}
    for fp in sorted(CKPT_DIR.glob("fold*.pt")):
        fi = int(fp.stem.replace("fold", ""))
        state = torch.load(str(fp), map_location="cpu", weights_only=False)
        m = FormantOffsetBiLSTM(cfg); m.load_state_dict(state["model_state"]); m.eval()
        folds[fi] = m; val_ids[fi] = state.get("val_ids", [])
    cid_to_fold = {c: fi for fi, ids in val_ids.items() for c in ids}

    # Cache per-clip (prod_tuples, gt, probs) once
    print("caching per-clip data...")
    cache = {}
    for cid in range(1, 41):
        feat_p = FORMANT_CACHE / f"vocadito_{cid}.npz"
        aud_p = VOC_AUDIO / f"vocadito_{cid}.wav"
        if not feat_p.exists() or not aud_p.exists(): continue
        gt = _gt_notes(cid)
        if not gt: continue
        fi = cid_to_fold.get(cid)
        if fi is None: continue
        y, sr = load_audio(str(aud_p), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
        prod = segment_pitch_to_notes(t, hz, vc, mc)
        prod_t = [(n.onset_s, n.offset_s, n.midi()) for n in prod]
        feat = np.load(feat_p)["mel"].astype(np.float32)
        probs = _probs(folds[fi], feat)
        cache[cid] = (prod_t, gt, probs)
        if cid % 10 == 0:
            print(f"  cached {cid}/40")
    print(f"cached {len(cache)} clips")

    # Sweep
    grid_min_prob = [0.3, 0.4, 0.5, 0.6, 0.7]
    grid_search_ms = [30, 50, 75, 100, 150]
    rows = []
    print(f"\n{'min_prob':<10}{'search_ms':<12}{'F-2e off20':<15}{'Δ vs prod':<12}{'win/lose/same'}")
    prod_means = []
    for cid, (pt, gt, probs) in cache.items():
        prod_means.append(_f1(pt, gt))
    prod_mean = float(np.mean(prod_means))

    for mp_th in grid_min_prob:
        for sm in grid_search_ms:
            f2e_means = []
            wins = losses = sames = 0
            for cid, (pt, gt, probs) in cache.items():
                pf1 = _f1(pt, gt)
                f2e_t = []
                for on, off, midi in pt:
                    new = _snap(off, probs, sm, mp_th)
                    if new - on < 0.05: new = on + 0.05
                    f2e_t.append((on, new, midi))
                ff1 = _f1(f2e_t, gt)
                f2e_means.append(ff1)
                if ff1 > pf1: wins += 1
                elif ff1 < pf1: losses += 1
                else: sames += 1
            f_mean = float(np.mean(f2e_means))
            d = f_mean - prod_mean
            rows.append({"min_prob": mp_th, "search_ms": sm,
                          "f2e_off20": f_mean, "delta": d,
                          "wins": wins, "losses": losses, "sames": sames})
            print(f"{mp_th:<10.2f}{sm:<12d}{f_mean:<15.4f}{d:<+12.4f}{wins}/{losses}/{sames}")

    rows.sort(key=lambda r: -r["delta"])
    print(f"\nProd baseline: {prod_mean:.4f}")
    print(f"Best config: min_prob={rows[0]['min_prob']} search_ms={rows[0]['search_ms']} "
          f"→ F-2e={rows[0]['f2e_off20']:.4f} (Δ {rows[0]['delta']:+.4f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"prod_off20": prod_mean, "sweep": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
