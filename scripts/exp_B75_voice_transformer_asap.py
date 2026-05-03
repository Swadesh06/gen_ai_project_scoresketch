"""B75 — Transformer voice tracker for ASAP Romantic (Phase D).

Addresses the Liszt 0.078 / Schumann 0.846 / Beethoven 0.897 voice-tracking
bottleneck: the greedy adaptive_pj tracker (B49) hits a ceiling on dense
chordal Romantic textures because it can't model long-range voice continuity.

Approach:
- ASAP piano scores have 2 PrettyMIDI instrument tracks (left + right hand)
  → binary voice supervision per note.
- Train a small Transformer over (pitch, onset_s, duration_s, position) →
  voice id (0 = left hand, 1 = right hand).
- At inference, replace `quantize_with_voice_tracking`'s greedy assigner
  with this learned model for Romantic-detected pieces.

Training set: aggregate notes from multiple ASAP pieces. Bach Fugues
have >2 voices but ASAP MIDI scores have 2 hands × interleaved voices.
For now, treat as binary L/R and see if it helps Liszt.

Pass criterion: Liszt snap > 0.10 (vs current 0.053 with YMT3+).
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
OUT_JSON = Path("reports/_exp_B75_voice_transformer.json")

# Train on a mix; eval on Liszt + Beethoven + Schumann + Chopin
TRAIN_PIECES = [
    "Bach/Fugue/bwv_846", "Bach/Fugue/bwv_848",
    "Bach/Fugue/bwv_854", "Bach/Fugue/bwv_856",
    "Beethoven/Sonata_5/1", "Beethoven/Sonata_18/2",
    "Schubert/Impromptu_op_90/1", "Schubert/Impromptu_op_90/3",
    "Mozart/Piano_Sonatas/8-1", "Mozart/Piano_Sonatas/11-1",
    "Brahms/Six_Pieces_op_118/2", "Mendelssohn/Songs_Without_Words/op_19_no_1",
]
VAL_PIECES = [
    "Liszt/Sonata", "Beethoven/Sonata_21/1",
    "Schumann/Toccata", "Chopin/Berceuse_Op_57",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def find_score_mid(piece: str) -> Path | None:
    """ASAP score MIDIs are at <piece_dir>/midi_score.mid."""
    p = ASAP / piece / "midi_score.mid"
    if p.exists():
        return p
    # Try common variants
    for var in [piece.replace("/Sonata", "/Sonata_21"), piece + "_no_1"]:
        q = ASAP / var / "midi_score.mid"
        if q.exists():
            return q
    return None


def load_piece(piece: str):
    """Load a piece, return (notes_array, voice_labels) where notes is
    (n, 4) of [midi_pitch, onset_s, duration_s, time_position] and
    voice_labels is (n,) ints (0 = first instrument = lower hand)."""
    mid_path = find_score_mid(piece)
    if mid_path is None:
        return None
    mid = pretty_midi.PrettyMIDI(str(mid_path))
    if len(mid.instruments) < 2:
        return None
    all_notes = []
    voices = []
    for vid, inst in enumerate(mid.instruments[:2]):
        for n in inst.notes:
            all_notes.append([float(n.pitch), float(n.start),
                              float(n.end - n.start), float(n.start)])
            voices.append(vid)
    arr = np.array(all_notes, dtype=np.float32)
    vc = np.array(voices, dtype=np.int64)
    # Sort by onset
    order = np.argsort(arr[:, 1])
    return arr[order], vc[order]


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
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 ff_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.feat_proj = nn.Linear(4, d_model)  # midi, onset, duration, time
        self.posenc = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                             dim_feedforward=ff_dim, dropout=0.1,
                                             batch_first=True, activation="gelu",
                                             norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.feat_proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        return self.head(h)  # (B, T, 2)


def chunk_piece(arr, vc, chunk_size=512):
    """Split into chunks for Transformer (max 512 notes / chunk for memory)."""
    n = len(arr)
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append((arr[i:i+chunk_size], vc[i:i+chunk_size]))
    return chunks


def normalise(arr):
    """Normalise note features for transformer input."""
    out = arr.copy()
    out[:, 0] = (out[:, 0] - 60) / 12  # MIDI -> octaves from C4
    # Make onset/duration/time relative to chunk start
    if len(out) > 0:
        out[:, 1] = out[:, 1] - out[0, 1]
        out[:, 3] = out[:, 3] - out[0, 3]
    return out


def main(n_epochs: int = 30, d_model: int = 128, n_layers: int = 4,
         lr: float = 3e-4, chunk_size: int = 512) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "d_model": d_model,
             "n_layers": n_layers, "lr": lr, "chunk_size": chunk_size,
             "task": "voice tracking 2-class L/R hand"}
    run = wandb.init(project="humscribe-v3.2", name="exp_B75_voice_transformer",
                     config=cfg_w, tags=["B75", "asap", "voice-tracker",
                                          "transformer", "phase-d"],
                     dir="logs/wandb")

    print("loading training pieces")
    train_data = []
    for p in TRAIN_PIECES:
        d = load_piece(p)
        if d is None:
            print(f"  skip {p}: no score MIDI / <2 voices")
            continue
        arr, vc = d
        for chunk_arr, chunk_vc in chunk_piece(arr, vc, chunk_size):
            if len(chunk_arr) >= 16:
                train_data.append((normalise(chunk_arr), chunk_vc))
        print(f"  {p}: {len(arr)} notes -> {sum(1 for _ in chunk_piece(arr, vc, chunk_size))} chunks")
    print(f"train chunks: {len(train_data)}")
    val_data = []
    for p in VAL_PIECES:
        d = load_piece(p)
        if d is None:
            print(f"  skip {p}: no score MIDI / <2 voices")
            continue
        arr, vc = d
        chunks = list(chunk_piece(arr, vc, chunk_size))
        for ch_arr, ch_vc in chunks:
            if len(ch_arr) >= 16:
                val_data.append((p, normalise(ch_arr), ch_vc))
        print(f"  val {p}: {len(arr)} notes -> {len(chunks)} chunks")
    print(f"val chunks: {len(val_data)}")

    if not train_data or not val_data:
        run.finish(); return

    model = VoiceTransformer(d_model=d_model, n_layers=n_layers).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    ce = nn.CrossEntropyLoss()
    print(f"\nmodel: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    best_acc = 0.0
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

        # Eval
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
        print(f"  ep {epoch:3d}  loss={train_loss:.3f}  val_acc={mean_acc:.4f}  best={best_acc:.4f}")
        for p, a in per_piece_acc.items():
            wandb.log({"epoch": epoch, f"acc_{p.split('/')[-1]}": a})
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                    "val_mean_acc": mean_acc, "best_acc": best_acc,
                    "lr": sched.get_last_lr()[0]})

    print(f"\nBest val mean acc: {best_acc:.4f}")
    print("Per-piece accuracy at best epoch:")
    for p, a in per_piece_acc.items():
        print(f"  {p}: {a:.4f}")
    wandb.summary["best_acc"] = best_acc
    wandb.summary["per_piece"] = per_piece_acc
    OUT_JSON.write_text(json.dumps({"best_acc": best_acc, "per_piece": per_piece_acc,
                                     "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk-size", type=int, default=512)
    main(**vars(ap.parse_args()))
