"""Phase F-2b: pretrain the formant offset detector on MIR-ST500.

Uses the 84 cached MIR-ST500 formant features at /workspace/.cache/mirst500_formant/
+ labels from MIR-ST500_corrected.json. Trains the F-2 architecture
(FormantOffsetBiLSTM h=96, l=2) and evaluates fine-tune on Vocadito.

Approach:
1. Window each MIR-ST500 song into 10-s clips. Discard windows with
   fewer than 5 notes (less informative).
2. Train the BiLSTM on the combined windowed dataset (200+ windows).
3. Evaluate on a held-out 5% of MIR-ST500 windows.
4. Optionally fine-tune on Vocadito (5-fold CV with the pretrained weights).

CPU-only training; OMP_NUM_THREADS=4 → uses 4 cores.
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


MIRST500_LABELS = Path("/workspace/datasets/mirst500/repo/MIR-ST500_20210206/MIR-ST500_corrected.json")
FORMANT_DIR = Path("/workspace/.cache/mirst500_formant")


def load_clip_windows(song_id: str, labels: list, win_s: float = 10.0,
                       hop_s: float = 0.01) -> list[tuple[np.ndarray, np.ndarray]]:
    """Window one song into per-10s (mel, offset-label) chunks."""
    f = FORMANT_DIR / f"{song_id}.npz"
    if not f.exists(): return []
    d = np.load(f)
    mel = d["mel"].astype(np.float32)  # (80, T)
    n_total = mel.shape[1]
    win_frames = int(win_s / hop_s)
    chunks = []
    for start in range(0, n_total - win_frames, win_frames):
        end = start + win_frames
        # Slice mel
        sub_mel = mel[:, start:end].T  # (T, 80)
        win_start_s = start * hop_s
        win_end_s = end * hop_s
        # Slice offsets that fall in this window
        offsets = [lbl[1] - win_start_s for lbl in labels
                    if win_start_s <= lbl[1] < win_end_s]
        if len(offsets) < 3: continue  # discard near-empty windows
        y = make_offset_labels(np.array(offsets), win_frames, hop_s=hop_s)
        chunks.append((sub_mel, y))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_phase_f_F2b_mirst500_pretrain.json"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/formant_offset_mirst500.pt"))
    args = ap.parse_args()

    print(f"loading labels from {MIRST500_LABELS}")
    with open(MIRST500_LABELS) as f:
        labels_all = json.load(f)
    print(f"  {len(labels_all)} songs in labels")

    available_ids = sorted([p.stem for p in FORMANT_DIR.glob("*.npz")],
                            key=lambda s: int(s))
    print(f"available cached features: {len(available_ids)} songs")
    print("windowing into 10s clips...")
    train_data = []
    test_data = []
    for sid in available_ids:
        if sid not in labels_all:
            continue
        wins = load_clip_windows(sid, labels_all[sid])
        # 95/5 train/test split per song
        n_test = max(1, len(wins) // 20)
        train_data.extend(wins[:-n_test] if n_test < len(wins) else wins)
        test_data.extend(wins[-n_test:] if n_test < len(wins) else [])
    print(f"  total windows: {len(train_data)} train, {len(test_data)} test")
    if len(train_data) < 50:
        print("not enough training windows"); return

    cfg = FormantOffsetConfig(in_dim=80, hidden=args.hidden,
                                layers=args.layers, dropout=0.2)
    model = FormantOffsetBiLSTM(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_w = torch.tensor([40.0])

    print(f"\ntraining {args.epochs} epochs on {len(train_data)} windows")
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0.0
        for x, y in train_data:
            xt = torch.from_numpy(x).unsqueeze(0)
            yt = torch.from_numpy(y).unsqueeze(0)
            logits = model(xt)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yt, pos_weight=pos_w)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item())
        avg = total_loss / len(train_data)
        if ep % 2 == 0:
            print(f"  ep {ep:3d}: train_loss={avg:.3f}")

    # Test eval
    model.eval()
    tps = fps = fns = 0
    with torch.no_grad():
        for x, y in test_data:
            probs = torch.sigmoid(model(torch.from_numpy(x).unsqueeze(0))).squeeze(0).numpy()
            pred = (probs > 0.5).astype(np.int32)
            diff = np.diff(np.concatenate([[0], pred]))
            true_diff = np.diff(np.concatenate([[0], y > 0.5]))
            starts = np.where(diff == 1)[0]
            true_starts = np.where(true_diff == 1)[0]
            matched = set()
            for s in starts:
                for ts in true_starts:
                    if abs(s - ts) <= 5 and ts not in matched:
                        matched.add(ts); break
            tps += len(matched); fps += len(starts) - len(matched)
            fns += len(true_starts) - len(matched)
    p = tps / max(tps + fps, 1)
    r = tps / max(tps + fns, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    print(f"\nMIR-ST500 held-out test F1={f1:.4f} (p={p:.3f}, r={r:.3f}, n_test={len(test_data)})")

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                  "test_f1": f1}, args.ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "epochs": args.epochs, "lr": args.lr,
        "hidden": args.hidden, "layers": args.layers,
        "n_train": len(train_data), "n_test": len(test_data),
        "test_f1": f1, "test_p": p, "test_r": r,
        "tps": tps, "fps": fps, "fns": fns,
    }, indent=2))
    print(f"saved checkpoint to {args.ckpt}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
