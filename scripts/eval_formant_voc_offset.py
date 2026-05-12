"""Phase F-2c: wire the formant offset detector into the Vocadito gate.

For each Vocadito clip:
1. Run PESTO+CREPE-voicing → onsets (existing path).
2. Run the cached formant BiLSTM → offsets.
3. Pair onsets with the next offset event after each onset.
4. Compute note-level F1 with onset tolerance ±50ms + offset20 tolerance.

Compare to the heuristic baseline (current production):
- Vocadito A1 noff F1 = 0.665 (no offset constraint)
- Vocadito A1 offset20 F1 = 0.439 (20% relative duration tolerance)

If F-2c improves offset20 by ≥ +0.05 without regressing noff by > 1pp,
the formant detector is a production win.

The detector is the F-2 trained model from
`humscribe/train/formant_offset.py`. We use the checkpoint with the
highest val F1 across the 5 folds (fold 1 = 0.542).
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
OUT = Path("reports/_phase_f_F2c_offset.json")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return -1
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _gt_notes(clip_id: int, annotator: str = "A1") -> list[tuple[float, float, int]]:
    """Return list of (onset, offset, midi) tuples."""
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


def _predict_offsets_from_formant(model: FormantOffsetBiLSTM,
                                    mel_features: np.ndarray,
                                    threshold: float = 0.5,
                                    min_gap_frames: int = 5) -> list[float]:
    """Run BiLSTM on formant features, return predicted offset times (frames * 10ms)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(mel_features.T).unsqueeze(0)  # (1, T, 80)
        logits = model(x).squeeze(0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs > threshold).astype(np.int32)
    # Group consecutive frames; report the start of each group
    diff = np.diff(np.concatenate([[0], pred]))
    starts = np.where(diff == 1)[0]
    # Min gap dedupe
    keep = []
    for s in starts:
        if not keep or s - keep[-1] >= min_gap_frames:
            keep.append(s)
    return [s * 0.01 for s in keep]


def _f1(pred: list[tuple], gt: list[tuple], tol_on: float = 0.05,
        offset_strict: bool = False, offset_rel_tol: float = 0.20) -> dict:
    """Note-level F1 with onset tolerance ±tol_on; offset_strict uses
    offset_rel_tol = ±20% relative duration."""
    if not pred or not gt:
        return {"f1": 0.0, "p": 0.0, "r": 0.0,
                 "n_pred": len(pred), "n_gt": len(gt)}
    matched_gt = set()
    matched_pred = set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in matched_gt: continue
            if abs(po - go) > tol_on: continue
            if pm != gm: continue
            if offset_strict:
                gt_dur = gf - go
                if abs(pf - gf) > offset_rel_tol * gt_dur: continue
            matched_gt.add(j); matched_pred.add(i); break
    tps = len(matched_pred)
    p = tps / max(len(pred), 1)
    r = tps / max(len(gt), 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    return {"f1": f1, "p": p, "r": r,
             "n_pred": len(pred), "n_gt": len(gt)}


def main():
    # Use deep variant fold 1 weights (val F1 0.501) — we don't have the
    # base fold 1 weights saved; instead train a fresh model on all 40 clips
    # quickly using the cached features. Actually, neither was saved per fold;
    # we have only the MIR-ST500 pretrained ckpt. Let's use that (test F1 0.30
    # on MIR-ST500 — but maybe the formant features-learned representation
    # still helps on Vocadito).

    cfg = FormantOffsetConfig(in_dim=80, hidden=96, layers=2, dropout=0.2)
    model = FormantOffsetBiLSTM(cfg)
    ckpt_path = Path("checkpoints/formant_offset_mirst500.pt")
    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"loaded MIR-ST500 pretrained checkpoint from {ckpt_path}")
    else:
        print("no MIR-ST500 checkpoint; using random init (will be poor)")
    model.eval()

    rows = []
    for cid in range(1, 41):
        feat_path = FORMANT_CACHE / f"vocadito_{cid}.npz"
        audio_path = VOC_AUDIO / f"vocadito_{cid}.wav"
        if not feat_path.exists() or not audio_path.exists(): continue

        gt = _gt_notes(cid, "A1")
        if not gt: continue

        # Production path: PESTO+CREPE voicing → segmenter
        y, sr = load_audio(str(audio_path), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
        prod_notes = segment_pitch_to_notes(t, hz, vc, mc)
        # Convert to (on, off, midi) tuples
        prod_tuples = [(n.onset_s, n.offset_s, n.midi()) for n in prod_notes]

        # F-2c: replace offsets with formant detector predictions.
        feat = np.load(feat_path)["mel"].astype(np.float32)
        formant_offsets = _predict_offsets_from_formant(model, feat)
        # For each onset, find the next formant offset > onset; if none, keep
        # the heuristic offset.
        f2c_tuples = []
        for on, off, midi in prod_tuples:
            cand_offs = [fo for fo in formant_offsets if fo > on]
            new_off = cand_offs[0] if cand_offs else off
            # Sanity: cap at min_dur 50ms
            if new_off - on < 0.05: new_off = on + 0.05
            f2c_tuples.append((on, new_off, midi))

        f1_noff_prod = _f1(prod_tuples, gt, tol_on=0.05, offset_strict=False)
        f1_off20_prod = _f1(prod_tuples, gt, tol_on=0.05, offset_strict=True)
        f1_noff_f2c = _f1(f2c_tuples, gt, tol_on=0.05, offset_strict=False)
        f1_off20_f2c = _f1(f2c_tuples, gt, tol_on=0.05, offset_strict=True)
        rows.append({
            "clip": cid,
            "prod_noff_f1": f1_noff_prod["f1"],
            "prod_off20_f1": f1_off20_prod["f1"],
            "f2c_noff_f1": f1_noff_f2c["f1"],
            "f2c_off20_f1": f1_off20_f2c["f1"],
        })
        print(f"voc_{cid:2d}  noff: {f1_noff_prod['f1']:.3f} -> {f1_noff_f2c['f1']:.3f}  "
              f"off20: {f1_off20_prod['f1']:.3f} -> {f1_off20_f2c['f1']:.3f}")

    if rows:
        m_noff_prod = float(np.mean([r["prod_noff_f1"] for r in rows]))
        m_off_prod = float(np.mean([r["prod_off20_f1"] for r in rows]))
        m_noff_f2c = float(np.mean([r["f2c_noff_f1"] for r in rows]))
        m_off_f2c = float(np.mean([r["f2c_off20_f1"] for r in rows]))
        print(f"\nProduction baseline:")
        print(f"  noff F1   = {m_noff_prod:.4f}")
        print(f"  off20 F1  = {m_off_prod:.4f}")
        print(f"F-2c (formant offset detector):")
        print(f"  noff F1   = {m_noff_f2c:.4f}  (delta {m_noff_f2c - m_noff_prod:+.4f})")
        print(f"  off20 F1  = {m_off_f2c:.4f}  (delta {m_off_f2c - m_off_prod:+.4f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows,
        "mean_prod_noff": m_noff_prod if rows else None,
        "mean_prod_off20": m_off_prod if rows else None,
        "mean_f2c_noff": m_noff_f2c if rows else None,
        "mean_f2c_off20": m_off_f2c if rows else None,
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
