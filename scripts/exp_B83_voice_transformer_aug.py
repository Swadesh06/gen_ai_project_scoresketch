"""B83 — B76 voice tracker with heavy MIDI augmentation (Phase D).

B76 hit 94.5% mean acc on 237 ASAP pieces. B83 adds online augmentation
to push past 95% (especially on Liszt 90.8% → ?):
- Pitch shift ±5 semitones (random per chunk)
- Time stretch 0.85-1.15× via index dilation
- Voice swap with probability 0.5 (data symmetry — L hand vs R hand is arbitrary label)
- Note dropout p=0.1
- Onset jitter ±15ms
"""
from __future__ import annotations
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import wandb

ASAP = Path("~/datasets/asap").expanduser()
OUT_JSON = Path("reports/_exp_B83_voice_transformer_aug.json")
CHECKPOINT = Path("checkpoints/voice_transformer_b83/best.pt")
HELD_OUT = {
    "Liszt/Sonata", "Schumann/Toccata", "Chopin/Berceuse_op_57",
    "Beethoven/Piano_Sonatas/21-1",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def all_pieces():
    return sorted(p.parent.relative_to(ASAP).as_posix()
                   for p in ASAP.rglob("midi_score.mid"))


def load_piece(piece: str):
    p = ASAP / piece / "midi_score.mid"
    if not p.exists():
        return None
    try:
        mid = pretty_midi.PrettyMIDI(str(p))
    except Exception:
        return None
    if len(mid.instruments) < 2:
        return None
    notes, voices = [], []
    for vid, inst in enumerate(mid.instruments[:2]):
        for n in inst.notes:
            notes.append([float(n.pitch), float(n.start),
                           float(n.end - n.start), float(n.start)])
            voices.append(vid)
    if not notes:
        return None
    arr = np.array(notes, dtype=np.float32)
    vc = np.array(voices, dtype=np.int64)
    order = np.argsort(arr[:, 1])
    return arr[order], vc[order]


def chunk_piece(arr, vc, chunk_size=512):
    out = []
    for i in range(0, len(arr), chunk_size):
        a, v = arr[i:i+chunk_size], vc[i:i+chunk_size]
        if len(a) >= 16:
            out.append((a, v))
    return out


def normalise(arr):
    out = arr.copy()
    out[:, 0] = (out[:, 0] - 60.0) / 12.0
    if len(out) > 0:
        out[:, 1] = out[:, 1] - out[0, 1]
        out[:, 3] = out[:, 3] - out[0, 3]
    return out


def augment(arr, vc, rng):
    a = arr.copy()
    v = vc.copy()
    # Pitch shift ±5 semitones
    shift = rng.uniform(-5, 5)
    a[:, 0] += shift
    # Time stretch via dilation
    stretch = rng.uniform(0.85, 1.15)
    a[:, 1] *= stretch
    a[:, 2] *= stretch
    a[:, 3] *= stretch
    # Voice swap probability 0.5 (label symmetry)
    if rng.random() < 0.5:
        v = 1 - v
    # Note dropout p=0.1
    keep = rng.random(len(a)) > 0.1
    if keep.sum() >= 16:
        a, v = a[keep], v[keep]
    # Onset jitter ±15ms
    jitter = rng.normal(0, 0.015, size=len(a))
    a[:, 1] += jitter
    a[:, 3] += jitter
    # Re-sort by onset
    order = np.argsort(a[:, 1])
    return a[order], v[order]


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=30000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class VoiceTransformer(nn.Module):
    def __init__(self, d_model=192, n_heads=6, n_layers=6, ff_dim=384):
        super().__init__()
        self.feat_proj = nn.Linear(4, d_model)
        self.posenc = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                             dim_feedforward=ff_dim, dropout=0.1,
                                             batch_first=True, activation="gelu",
                                             norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)
    def forward(self, x):
        h = self.feat_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        return self.head(h)


def main(n_epochs: int = 60, d_model: int = 192, n_layers: int = 6,
         lr: float = 3e-4, chunk_size: int = 512,
         n_aug_per_chunk: int = 2) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "d_model": d_model,
             "n_layers": n_layers, "lr": lr, "chunk_size": chunk_size,
             "n_aug_per_chunk": n_aug_per_chunk}
    run = wandb.init(project="humscribe-v3.2", name="exp_B83_voice_transformer_aug",
                     config=cfg_w, tags=["B83", "asap", "voice-tracker", "augment",
                                          "phase-d"], dir="logs/wandb")
    print("scanning ASAP pieces…")
    pieces = all_pieces()
    train_data, val_data = [], []
    for p in pieces:
        d = load_piece(p)
        if d is None: continue
        arr, vc = d
        chs = chunk_piece(arr, vc, chunk_size)
        for ch_arr, ch_vc in chs:
            if p in HELD_OUT:
                val_data.append((p, normalise(ch_arr), ch_vc))
            else:
                train_data.append((normalise(ch_arr), ch_vc))
    print(f"  train chunks: {len(train_data)}; val chunks: {len(val_data)}")

    model = VoiceTransformer(d_model=d_model, n_layers=n_layers).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    ce = nn.CrossEntropyLoss()
    rng = np.random.RandomState(42)
    print(f"  model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_per_piece = {}
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_data)
        losses = []
        for arr, vc in train_data:
            # Original
            samples = [(arr, vc)]
            # n_aug_per_chunk augmented copies
            for _ in range(n_aug_per_chunk):
                samples.append(augment(arr, vc, rng))
            for a, v in samples:
                if len(a) < 16: continue
                x = torch.from_numpy(a).unsqueeze(0).to("cuda")
                y = torch.from_numpy(v).unsqueeze(0).to("cuda")
                opt.zero_grad()
                logits = model(x)
                loss = ce(logits.view(-1, 2), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
        sched.step()
        train_loss = sum(losses) / len(losses)

        model.eval()
        per_piece = {}
        with torch.no_grad():
            for piece, arr, vc in val_data:
                x = torch.from_numpy(arr).unsqueeze(0).to("cuda")
                pred = model(x).argmax(-1).squeeze(0).cpu().numpy()
                acc = float((pred == vc).mean())
                per_piece.setdefault(piece, []).append(acc)
        per_piece_acc = {p: float(np.mean(a)) for p, a in per_piece.items()}
        mean_acc = float(np.mean(list(per_piece_acc.values())))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_per_piece = dict(per_piece_acc)
            torch.save({"model": model.state_dict(), "config": cfg_w,
                        "best_acc": best_acc, "per_piece": per_piece_acc}, str(CHECKPOINT))
        print(f"  ep {epoch:3d}  loss={train_loss:.3f}  val_mean_acc={mean_acc:.4f}  best={best_acc:.4f}")
        for p, a in per_piece_acc.items():
            wandb.log({"epoch": epoch, f"acc_{p.replace('/', '_')}": a})
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                   "val_mean_acc": mean_acc, "best_acc": best_acc,
                   "lr": sched.get_last_lr()[0]})

    print(f"\nBest val mean acc: {best_acc:.4f}")
    print("Per-piece accuracy at best epoch:")
    for p, a in best_per_piece.items():
        print(f"  {p}: {a:.4f}")
    wandb.summary.update({"best_acc": best_acc, "best_per_piece": best_per_piece})
    OUT_JSON.write_text(json.dumps({"best_acc": best_acc, "best_per_piece": best_per_piece,
                                     "config": cfg_w}, indent=2))
    print(f"  saved best to {CHECKPOINT}")
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--n-aug-per-chunk", type=int, default=2)
    main(**vars(ap.parse_args()))
