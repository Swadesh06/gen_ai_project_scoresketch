"""Phase F-2: train the formant-band offset detector.

Tiny BiLSTM on the cached 80-bin formant mel-spectrogram, 5-fold CV
on 40 Vocadito clips. Targets the offset20 gap.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.train.formant_offset import (
    FormantOffsetBiLSTM, FormantOffsetConfig, make_offset_labels,
)

FORMANT_DIR = Path("/workspace/.cache/voc_formant")
VOC_NOTES = Path("/home/swadesh/datasets/vocadito/Annotations/Notes")


def load_clip(clip_id: int, annotator: str = "A1") -> tuple[np.ndarray, np.ndarray] | None:
    f = FORMANT_DIR / f"vocadito_{clip_id}.npz"
    csv = VOC_NOTES / f"vocadito_{clip_id}_notes{annotator}.csv"
    if not f.exists() or not csv.exists():
        return None
    d = np.load(f)
    mel = d["mel"].astype(np.float32)  # (80, T)
    # Parse offsets: each line is (onset, freq, duration), offset = onset+dur
    offset_times = []
    for line in csv.read_text().splitlines():
        if not line.strip(): continue
        a, b, c = line.split(",")
        offset_times.append(float(a) + float(c))
    n_frames = mel.shape[1]
    y = make_offset_labels(np.array(offset_times), n_frames, hop_s=0.01)
    return mel.T, y  # (T, 80), (T,)


def train_one_fold(train_ids: list[int], val_ids: list[int],
                    cfg: FormantOffsetConfig, epochs: int = 30,
                    lr: float = 1e-3, device: str = "cpu",
                    save_path: Path | None = None) -> dict:
    """Train BiLSTM on train_ids, return val F1 on val_ids.

    If save_path is given, the trained weights are saved to that path
    for later production use (F-2c/F-2d wiring into segmenter).
    """
    model = FormantOffsetBiLSTM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_data = []
    for cid in train_ids:
        r = load_clip(cid, "A1")
        if r is None: continue
        train_data.append(r)
    if not train_data:
        return {"val_f1": 0.0, "n_train": 0}

    for ep in range(epochs):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0.0
        for x, y in train_data:
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            yt = torch.from_numpy(y).unsqueeze(0).to(device)
            logits = model(xt)
            pos_w = torch.tensor([40.0]).to(device)  # class imbalance
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yt, pos_weight=pos_w,
            )
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item())
        if ep % 5 == 0:
            print(f"  ep {ep}: train_loss={total_loss/len(train_data):.3f}")

    # Validate: F1 at threshold 0.5
    model.eval()
    tps = fps = fns = 0
    for cid in val_ids:
        r = load_clip(cid, "A1")
        if r is None: continue
        x, y = r
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(x).unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
        pred = (probs > 0.5).astype(np.int32)
        # Group consecutive predictions as single events
        diff = np.diff(np.concatenate([[0], pred]))
        starts = np.where(diff == 1)[0]
        # Compare against y events similarly
        true_diff = np.diff(np.concatenate([[0], y > 0.5]))
        true_starts = np.where(true_diff == 1)[0]
        # F1 by frame-tolerance ±5 frames (50ms)
        matched = set()
        for s in starts:
            for ts in true_starts:
                if abs(s - ts) <= 5 and ts not in matched:
                    matched.add(ts); break
        tps += len(matched)
        fps += len(starts) - len(matched)
        fns += len(true_starts) - len(matched)
    p = tps / max(tps + fps, 1)
    r = tps / max(tps + fns, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    print(f"  val F1 = {f1:.3f}  (p={p:.3f}, r={r:.3f}, tps={tps}, fps={fps}, fns={fns})")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                      "val_f1": f1, "val_ids": list(val_ids)}, save_path)
        print(f"  saved fold ckpt: {save_path}")
    return {"val_f1": f1, "val_p": p, "val_r": r, "tps": tps, "fps": fps, "fns": fns,
             "n_train": len(train_data), "n_val": len(val_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_phase_f_F2_formant.json"))
    args = ap.parse_args()
    device = "cpu"  # small model, CPU is enough; keeps GPU free for C5b

    cfg = FormantOffsetConfig(in_dim=80, hidden=96, layers=2, dropout=0.2)
    ids = list(range(1, args.n_clips + 1))
    np.random.seed(0)
    np.random.shuffle(ids)
    fold_size = len(ids) // args.folds
    fold_results = []
    ckpt_root = Path("checkpoints/formant_offset_vocadito")
    for fi in range(args.folds):
        val = ids[fi * fold_size: (fi + 1) * fold_size]
        train = [i for i in ids if i not in val]
        print(f"\n=== fold {fi+1}/{args.folds}  train={len(train)} val={len(val)} ===")
        save_path = ckpt_root / f"fold{fi}.pt"
        res = train_one_fold(train, val, cfg, epochs=args.epochs,
                              device=device, save_path=save_path)
        fold_results.append({"fold": fi, "val_ids": val, **res})
    mean_f1 = float(np.mean([r["val_f1"] for r in fold_results]))
    print(f"\n5-fold mean offset F1 = {mean_f1:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"folds": fold_results,
                                      "mean_f1": mean_f1}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
