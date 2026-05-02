"""Threshold sweep over an already-trained BiLSTM onset model. No re-training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
import torch

from humscribe.audio_io import load_audio
from humscribe.notes import midi_to_hz
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.train.onset_bilstm import (
    OnsetBiLSTM, OnsetModelConfig, predict_onsets, segment_via_onsets,
)


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def main(ckpt_path: str, annotator: str) -> None:
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = OnsetModelConfig(**ck["config"])
    model = OnsetBiLSTM(cfg)
    model.load_state_dict(ck["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    summary_data = json.loads(Path(ckpt_path.replace("checkpoints/", "reports/_exp_B10_").replace("/best.pt", ".json")).read_text())
    val_ids = summary_data.get("val_ids", [])
    if not val_ids:
        print("no val_ids in summary; falling back to all clips")
        val_ids = sorted(p.stem for p in (VOC / "Audio").glob("*.wav"))

    print(f"evaluating on {len(val_ids)} val clips")

    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for cid in val_ids:
        wav = VOC / "Audio" / f"{cid}.wav"
        nf = VOC / "Annotations" / "Notes" / f"{cid}_notes{annotator}.csv"
        if not wav.exists() or not nf.exists():
            continue
        audio, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_pesto(audio, sr)
        gt_iv, gt_p = load_notes(nf)
        cache[cid] = (t, hz, vc, gt_iv, gt_p)

    print(f"\n{'thresh':>7s}  {'mean_F1':>7s}  {'mean_P':>7s}  {'mean_R':>7s}")
    best = (-1.0, 0.5, 0.0, 0.0)
    for thr in (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
        f1s, ps, rs = [], [], []
        for cid, (t, hz, vc, gt_iv, gt_p) in cache.items():
            mask = predict_onsets(model, hz, vc, threshold=thr, device=device)
            notes = segment_via_onsets(t, hz, vc, mask)
            if not notes:
                f1s.append(0.0); ps.append(0.0); rs.append(0.0); continue
            eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
            eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                gt_iv, gt_p, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
            )
            f1s.append(float(f)); ps.append(float(p)); rs.append(float(r))
        mf1 = float(np.mean(f1s)); mp = float(np.mean(ps)); mr = float(np.mean(rs))
        print(f"  {thr:.3f}    {mf1:.3f}    {mp:.3f}    {mr:.3f}")
        if mf1 > best[0]:
            best = (mf1, thr, mp, mr)
    print(f"\nbest: F1={best[0]:.3f}  P={best[2]:.3f}  R={best[3]:.3f}  at threshold={best[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/onset_bilstm_seed0_h64/best.pt")
    ap.add_argument("--annotator", default="A1")
    args = ap.parse_args()
    main(args.ckpt, args.annotator)
