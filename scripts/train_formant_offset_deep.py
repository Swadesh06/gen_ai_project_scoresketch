"""Phase F-2 follow-up: deeper formant offset detector (hidden=128, layers=3)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humscribe.train.formant_offset import FormantOffsetConfig
from scripts.train_formant_offset import main as base_main, train_one_fold, load_clip
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_phase_f_F2_formant_deep.json"))
    args = ap.parse_args()

    cfg = FormantOffsetConfig(in_dim=80, hidden=args.hidden,
                                layers=args.layers, dropout=0.3)
    ids = list(range(1, args.n_clips + 1))
    np.random.seed(0)
    np.random.shuffle(ids)
    fold_size = len(ids) // args.folds
    fold_results = []
    for fi in range(args.folds):
        val = ids[fi * fold_size: (fi + 1) * fold_size]
        train = [i for i in ids if i not in val]
        print(f"\n=== fold {fi+1}/{args.folds}  hidden={args.hidden} layers={args.layers}  train={len(train)} val={len(val)} ===")
        res = train_one_fold(train, val, cfg, epochs=args.epochs, device="cpu")
        fold_results.append({"fold": fi, **res})
    mean_f1 = float(np.mean([r["val_f1"] for r in fold_results]))
    print(f"\n5-fold mean offset F1 (deep) = {mean_f1:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"hidden": args.hidden, "layers": args.layers,
                                      "folds": fold_results, "mean_f1": mean_f1},
                                     indent=2))


if __name__ == "__main__":
    main()
