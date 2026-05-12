"""Phase F-2d: production wiring using Vocadito-fold-trained weights.

For each Vocadito clip, identify which fold's val set it belongs to
(np.random.seed(0) + shuffle + 5-fold split). Load that fold's
checkpoint, predict offsets, score.

This is the correct cross-validated F-2c. Replaces F-2c (which used
the wrong-domain MIR-ST500 weights).
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
OUT = Path("reports/_phase_f_F2d_offset.json")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0: return -1
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _gt_notes(clip_id: int, annotator: str = "A1") -> list[tuple[float, float, int]]:
    csv_path = VOC_NOTES / f"vocadito_{clip_id}_notes{annotator}.csv"
    out = []
    if not csv_path.exists(): return out
    for line in csv_path.read_text().splitlines():
        if not line.strip(): continue
        a, b, c = line.split(",")
        on = float(a); freq = float(b); dur = float(c)
        midi = _hz_to_midi(freq)
        if midi >= 1:
            out.append((on, on + dur, midi))
    return out


def _predict_offsets(model: FormantOffsetBiLSTM, mel: np.ndarray,
                      threshold: float = 0.5, min_gap_frames: int = 5) -> list[float]:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(mel.T).unsqueeze(0)
        logits = model(x).squeeze(0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs > threshold).astype(np.int32)
    diff = np.diff(np.concatenate([[0], pred]))
    starts = np.where(diff == 1)[0]
    keep = []
    for s in starts:
        if not keep or s - keep[-1] >= min_gap_frames:
            keep.append(s)
    return [s * 0.01 for s in keep]


def _f1(pred: list[tuple], gt: list[tuple], tol_on: float = 0.05,
         offset_strict: bool = False, offset_rel_tol: float = 0.20) -> dict:
    if not pred or not gt:
        return {"f1": 0.0, "n_pred": len(pred), "n_gt": len(gt)}
    matched_gt = set(); matched_pred = set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in matched_gt: continue
            if abs(po - go) > tol_on: continue
            if pm != gm: continue
            if offset_strict:
                if abs(pf - gf) > offset_rel_tol * (gf - go): continue
            matched_gt.add(j); matched_pred.add(i); break
    tps = len(matched_pred)
    p = tps / max(len(pred), 1); r = tps / max(len(gt), 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    return {"f1": f1, "p": p, "r": r, "n_pred": len(pred), "n_gt": len(gt)}


def _load_fold_models() -> dict[int, FormantOffsetBiLSTM]:
    cfg = FormantOffsetConfig(in_dim=80, hidden=96, layers=2, dropout=0.2)
    folds = {}
    for fp in sorted(CKPT_DIR.glob("fold*.pt")):
        fi = int(fp.stem.replace("fold", ""))
        m = FormantOffsetBiLSTM(cfg)
        state = torch.load(str(fp), map_location="cpu", weights_only=False)
        m.load_state_dict(state["model_state"])
        m.eval()
        folds[fi] = m
        print(f"  loaded fold {fi}: val_ids={state.get('val_ids', [])[:5]}...  val_f1={state.get('val_f1'):.3f}")
    return folds


def _val_id_to_fold(val_ids_per_fold: dict[int, list[int]]) -> dict[int, int]:
    """clip_id → fold index that held this clip out for validation."""
    m = {}
    for fi, ids in val_ids_per_fold.items():
        for cid in ids:
            m[cid] = fi
    return m


def main():
    folds = _load_fold_models()
    if not folds:
        print(f"no fold checkpoints at {CKPT_DIR}"); return
    val_ids = {}
    for fp in sorted(CKPT_DIR.glob("fold*.pt")):
        fi = int(fp.stem.replace("fold", ""))
        state = torch.load(str(fp), map_location="cpu", weights_only=False)
        val_ids[fi] = state.get("val_ids", [])
    cid_to_fold = _val_id_to_fold(val_ids)
    rows = []
    for cid in range(1, 41):
        feat_path = FORMANT_CACHE / f"vocadito_{cid}.npz"
        audio_path = VOC_AUDIO / f"vocadito_{cid}.wav"
        if not feat_path.exists() or not audio_path.exists(): continue
        gt = _gt_notes(cid, "A1")
        if not gt: continue
        fi = cid_to_fold.get(cid)
        if fi is None or fi not in folds:
            print(f"voc_{cid}: no held-out fold; skipping"); continue
        model = folds[fi]
        y, sr = load_audio(str(audio_path), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
        prod = segment_pitch_to_notes(t, hz, vc, mc)
        prod_tuples = [(n.onset_s, n.offset_s, n.midi()) for n in prod]

        feat = np.load(feat_path)["mel"].astype(np.float32)
        form_offsets = _predict_offsets(model, feat)
        f2d_tuples = []
        for on, off, midi in prod_tuples:
            cand = [fo for fo in form_offsets if fo > on]
            new_off = cand[0] if cand else off
            if new_off - on < 0.05: new_off = on + 0.05
            f2d_tuples.append((on, new_off, midi))

        prod_noff = _f1(prod_tuples, gt, offset_strict=False)["f1"]
        prod_off20 = _f1(prod_tuples, gt, offset_strict=True)["f1"]
        f2d_noff = _f1(f2d_tuples, gt, offset_strict=False)["f1"]
        f2d_off20 = _f1(f2d_tuples, gt, offset_strict=True)["f1"]
        rows.append({"clip": cid, "fold": fi,
                      "prod_noff_f1": prod_noff, "prod_off20_f1": prod_off20,
                      "f2d_noff_f1": f2d_noff, "f2d_off20_f1": f2d_off20})
        print(f"voc_{cid:2d} fold={fi}  noff: {prod_noff:.3f}->{f2d_noff:.3f}  "
              f"off20: {prod_off20:.3f}->{f2d_off20:.3f}")
    if rows:
        m_pn = float(np.mean([r["prod_noff_f1"] for r in rows]))
        m_po = float(np.mean([r["prod_off20_f1"] for r in rows]))
        m_fn = float(np.mean([r["f2d_noff_f1"] for r in rows]))
        m_fo = float(np.mean([r["f2d_off20_f1"] for r in rows]))
        print(f"\nMean noff:  prod {m_pn:.4f}  F-2d {m_fn:.4f}  (Δ {m_fn-m_pn:+.4f})")
        print(f"Mean off20: prod {m_po:.4f}  F-2d {m_fo:.4f}  (Δ {m_fo-m_po:+.4f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows,
                                "mean_prod_noff": m_pn if rows else None,
                                "mean_prod_off20": m_po if rows else None,
                                "mean_f2d_noff": m_fn if rows else None,
                                "mean_f2d_off20": m_fo if rows else None,
                               }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
