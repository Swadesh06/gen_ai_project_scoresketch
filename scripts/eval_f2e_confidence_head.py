"""Phase F-2e: BiLSTM as confidence head on top of heuristic offsets.

F-2c/F-2d showed that *replacing* the heuristic offset with the BiLSTM
prediction regresses offset20-F1. Reason: the BiLSTM's offset prediction
is event-level F1 ~0.47, fine for "did it fire near a real offset" but
~50 ms timing error is too coarse for the ±20% duration tolerance.

F-2e tries the gentler approach: take the heuristic offset, then snap
it to the nearest BiLSTM peak within a search window. This combines
the heuristic's accurate timing (when voicing is unambiguous) with
the BiLSTM's correction (when voicing is noisy / vibrato).
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
from humscribe.notes import NoteEvent
from humscribe.train.formant_offset import (
    FormantOffsetBiLSTM, FormantOffsetConfig,
)
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC_AUDIO = Path("/home/swadesh/datasets/vocadito/Audio")
VOC_NOTES = Path("/home/swadesh/datasets/vocadito/Annotations/Notes")
FORMANT_CACHE = Path("/workspace/.cache/voc_formant")
CKPT_DIR = Path("checkpoints/formant_offset_vocadito")
OUT = Path("reports/_phase_f_F2e_offset.json")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0: return -1
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _gt_notes(clip_id: int) -> list[tuple[float, float, int]]:
    csv = VOC_NOTES / f"vocadito_{clip_id}_notesA1.csv"
    out = []
    if not csv.exists(): return out
    for line in csv.read_text().splitlines():
        if not line.strip(): continue
        a, b, c = line.split(",")
        on = float(a); freq = float(b); dur = float(c)
        midi = _hz_to_midi(freq)
        if midi >= 1: out.append((on, on + dur, midi))
    return out


def _bilstm_probs(model: FormantOffsetBiLSTM, mel: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(mel.T).unsqueeze(0)
        logits = model(x).squeeze(0).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def _f1(pred, gt, tol_on=0.05, offset_strict=False, offset_rel_tol=0.20) -> float:
    if not pred or not gt: return 0.0
    matched_gt = set(); matched_pred = set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in matched_gt: continue
            if abs(po - go) > tol_on: continue
            if pm != gm: continue
            if offset_strict and abs(pf - gf) > offset_rel_tol * (gf - go):
                continue
            matched_gt.add(j); matched_pred.add(i); break
    tps = len(matched_pred)
    p = tps / max(len(pred), 1); r = tps / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-6)


def _confidence_head_snap(heur_off: float, probs: np.ndarray,
                            search_ms: float = 100.0,
                            hop_s: float = 0.01,
                            min_prob: float = 0.5) -> float:
    """Find the BiLSTM peak within ±search_ms of the heuristic offset.
    If peak > min_prob, return the peak time; else return the heuristic.
    """
    n = len(probs)
    center = int(heur_off / hop_s)
    window = int(search_ms / 1000.0 / hop_s)
    lo = max(0, center - window)
    hi = min(n, center + window + 1)
    if lo >= hi: return heur_off
    slice_probs = probs[lo:hi]
    peak_idx = int(np.argmax(slice_probs))
    peak_prob = float(slice_probs[peak_idx])
    if peak_prob < min_prob:
        return heur_off  # BiLSTM not confident → keep heuristic
    return (lo + peak_idx) * hop_s


def main():
    cfg = FormantOffsetConfig(in_dim=80, hidden=96, layers=2, dropout=0.2)
    val_ids = {}
    folds = {}
    for fp in sorted(CKPT_DIR.glob("fold*.pt")):
        fi = int(fp.stem.replace("fold", ""))
        state = torch.load(str(fp), map_location="cpu", weights_only=False)
        m = FormantOffsetBiLSTM(cfg)
        m.load_state_dict(state["model_state"])
        m.eval()
        folds[fi] = m
        val_ids[fi] = state.get("val_ids", [])
    cid_to_fold = {cid: fi for fi, ids in val_ids.items() for cid in ids}

    rows = []
    for cid in range(1, 41):
        feat_path = FORMANT_CACHE / f"vocadito_{cid}.npz"
        audio_path = VOC_AUDIO / f"vocadito_{cid}.wav"
        if not feat_path.exists() or not audio_path.exists(): continue
        gt = _gt_notes(cid)
        if not gt: continue
        fi = cid_to_fold.get(cid)
        if fi is None or fi not in folds: continue
        model = folds[fi]
        y, sr = load_audio(str(audio_path), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
        prod = segment_pitch_to_notes(t, hz, vc, mc)
        prod_tuples = [(n.onset_s, n.offset_s, n.midi()) for n in prod]
        feat = np.load(feat_path)["mel"].astype(np.float32)
        probs = _bilstm_probs(model, feat)

        f2e_tuples = []
        for on, off, midi in prod_tuples:
            new_off = _confidence_head_snap(off, probs)
            if new_off - on < 0.05: new_off = on + 0.05
            f2e_tuples.append((on, new_off, midi))

        prod_off20 = _f1(prod_tuples, gt, offset_strict=True)
        f2e_off20 = _f1(f2e_tuples, gt, offset_strict=True)
        rows.append({"clip": cid, "fold": fi,
                      "prod_off20": prod_off20, "f2e_off20": f2e_off20,
                      "delta": f2e_off20 - prod_off20})
        print(f"voc_{cid:2d} fold={fi}  off20: {prod_off20:.3f}->{f2e_off20:.3f}  "
              f"Δ={f2e_off20-prod_off20:+.3f}")

    if rows:
        m_prod = float(np.mean([r["prod_off20"] for r in rows]))
        m_f2e = float(np.mean([r["f2e_off20"] for r in rows]))
        print(f"\nMean off20: prod {m_prod:.4f}  F-2e {m_f2e:.4f}  (Δ {m_f2e-m_prod:+.4f})")
        n_win = sum(1 for r in rows if r["delta"] > 0)
        n_lose = sum(1 for r in rows if r["delta"] < 0)
        n_same = sum(1 for r in rows if r["delta"] == 0)
        print(f"win/lose/same: {n_win}/{n_lose}/{n_same}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows,
                                 "mean_prod_off20": m_prod if rows else None,
                                 "mean_f2e_off20": m_f2e if rows else None,
                                 "n_win": n_win, "n_lose": n_lose,
                                 "n_same": n_same}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
