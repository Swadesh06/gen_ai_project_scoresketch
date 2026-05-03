"""B76 — Scale up B75 voice tracker to all 242 ASAP pieces (Phase D).

B75 hit 76% Liszt / 82% Schumann val accuracy with only 12 train pieces.
ASAP has 242 pieces with midi_score.mid. Scaling up should push accuracy
into the 85-90% range that would actually move snap on Liszt.

Held-out test pieces: Liszt Sonata, Schumann Toccata, Chopin Berceuse,
Beethoven Piano_Sonatas/21-1 — the 4 Romantic pieces in the v2 spec
that suffer from voice-tracking failure.

Save best-by-val model so it can be used as a drop-in replacement for
the greedy adaptive_pj voice tracker (Phase D follow-up).
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
OUT_JSON = Path("reports/_exp_B76_voice_transformer_scaled.json")
CHECKPOINT = Path("checkpoints/voice_transformer_b76/best.pt")

# Held-out test pieces — the v2-spec Romantic pieces with worst snap
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
    """Return list of all ASAP pieces with midi_score.mid."""
    return sorted(p.parent.relative_to(ASAP).as_posix()
                   for p in ASAP.rglob("midi_score.mid"))


def load_piece(piece: str):
    mid_path = ASAP / piece / "midi_score.mid"
    if not mid_path.exists():
        return None
    try:
        mid = pretty_midi.PrettyMIDI(str(mid_path))
    except Exception:
        return None
    if len(mid.instruments) < 2:
        return None
    notes = []
    voices = []
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
    chunks = []
    for i in range(0, len(arr), chunk_size):
        a = arr[i:i+chunk_size]
        v = vc[i:i+chunk_size]
        if len(a) >= 16:
            chunks.append((a, v))
    return chunks


def normalise(arr):
    out = arr.copy()
    out[:, 0] = (out[:, 0] - 60) / 12
    if len(out) > 0:
        out[:, 1] = out[:, 1] - out[0, 1]
        out[:, 3] = out[:, 3] - out[0, 3]
    return out


class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 30000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VoiceTransformer(nn.Module):
    def __init__(self, d_model: int = 192, n_heads: int = 6, n_layers: int = 6,
                 ff_dim: int = 384):
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


def main(n_epochs: int = 50, d_model: int = 192, n_layers: int = 6,
         lr: float = 3e-4, chunk_size: int = 512) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "d_model": d_model,
             "n_layers": n_layers, "lr": lr, "chunk_size": chunk_size}
    run = wandb.init(project="humscribe-v3.2", name="exp_B76_voice_transformer_scaled",
                     config=cfg_w, tags=["B76", "asap", "voice-tracker",
                                          "transformer", "phase-d", "scaled"],
                     dir="logs/wandb")
    print("scanning ASAP pieces…")
    pieces = all_pieces()
    print(f"  total pieces: {len(pieces)}")
    train_data = []
    val_data = []
    skipped = 0
    for p in pieces:
        d = load_piece(p)
        if d is None:
            skipped += 1; continue
        arr, vc = d
        is_held_out = p in HELD_OUT
        for ch_arr, ch_vc in chunk_piece(arr, vc, chunk_size):
            if is_held_out:
                val_data.append((p, normalise(ch_arr), ch_vc))
            else:
                train_data.append((normalise(ch_arr), ch_vc))
    print(f"  skipped (no MIDI / <2 voices): {skipped}")
    print(f"  train chunks: {len(train_data)} from {len(pieces) - skipped - len(HELD_OUT)} pieces")
    print(f"  val chunks: {len(val_data)} from held-out: {sorted(HELD_OUT)}")

    if not train_data or not val_data:
        run.finish(); return

    model = VoiceTransformer(d_model=d_model, n_layers=n_layers).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    ce = nn.CrossEntropyLoss()
    print(f"  model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    wandb.summary["n_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    wandb.summary["n_train_chunks"] = len(train_data)
    wandb.summary["n_val_chunks"] = len(val_data)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_per_piece = {}
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(train_data)
        losses = []
        for arr, vc in train_data:
            x = torch.from_numpy(arr).unsqueeze(0).to("cuda")
            y = torch.from_numpy(vc).unsqueeze(0).to("cuda")
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

        print(f"  ep {epoch:3d}  loss={train_loss:.3f}  mean_val_acc={mean_acc:.4f}  best={best_acc:.4f}")
        for p, a in per_piece_acc.items():
            wandb.log({"epoch": epoch, f"acc_{p.replace('/', '_')}": a})
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                   "val_mean_acc": mean_acc, "best_acc": best_acc,
                   "lr": sched.get_last_lr()[0]})

    print(f"\nBest val mean acc: {best_acc:.4f}")
    print("Per-piece accuracy at best epoch:")
    for p, a in best_per_piece.items():
        print(f"  {p}: {a:.4f}")
    wandb.summary["best_acc"] = best_acc
    wandb.summary["best_per_piece"] = best_per_piece
    OUT_JSON.write_text(json.dumps({"best_acc": best_acc, "best_per_piece": best_per_piece,
                                     "n_train_chunks": len(train_data),
                                     "n_val_chunks": len(val_data),
                                     "config": cfg_w}, indent=2))
    print(f"  saved best to {CHECKPOINT}")
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk-size", type=int, default=512)
    main(**vars(ap.parse_args()))
