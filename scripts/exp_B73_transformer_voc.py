"""B73 — Transformer voicing/onset detector on Vocadito + MTG-QBH (Phase D).

Replaces the BiLSTM with a small Transformer encoder over the same feature
sequence (mel + PESTO + CREPE on 100Hz timebase). Transformers can in
principle capture long-range dependencies that LSTMs miss — vibrato has
a ~5Hz periodicity that spans 20 frames, which a Transformer's
self-attention should pick up directly.

Architecture: 4-layer Transformer encoder, hidden=192, 4 heads, sinusoidal
positional encoding, learnable [CLS]-style absent. Same training set as B70
(40 Vocadito + 118 MTG-QBH pseudo).

Pass criterion: 5-fold CV mean F1 ≥ 0.50 (this is meaningful even below
heuristic 0.665 — we want to know if the architecture helps at all).
"""
from __future__ import annotations
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import midi_to_hz
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes

VOC = Path("~/datasets/vocadito").expanduser()
QBH = Path("~/datasets/mtg_qbh/audio").expanduser()
CACHE = Path("/workspace/.cache/voc_qbh_features")
OUT_JSON = Path("reports/_exp_B73_transformer.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_vocadito_notes(annotation_csv: Path):
    rows = [r.strip().split(",") for r in annotation_csv.read_text().splitlines() if r.strip()]
    if not rows:
        return np.empty((0, 2)), np.empty(0)
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([onsets, onsets + durations], axis=1), pitches


def pseudo_label_clip(audio_path: str):
    audio, sr = load_audio(audio_path, target_sr=22050)
    t, hz, vc = track_pitch_hybrid_voicing(audio, sr)
    mode_cfg = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    notes = segment_pitch_to_notes(t, hz, vc, mode_cfg)
    notes = [n for n in notes if (n.offset_s - n.onset_s) >= mode_cfg.min_note_seconds]
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    pitches = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    return intervals, pitches


def build_targets(t: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    label = np.zeros(len(t), dtype=np.float32)
    for on, off in intervals:
        label[(t >= on) & (t <= off)] = 1.0
    return label


def load_or_extract(audio_path: str):
    cache = CACHE / f"{Path(audio_path).stem}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    raise FileNotFoundError(f"feature cache missing for {audio_path}; run B70 first")


class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerOnset(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 192, n_heads: int = 4,
                 n_layers: int = 4, ff_dim: int = 384):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.posenc = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                             dim_feedforward=ff_dim,
                                             dropout=0.2, batch_first=True,
                                             activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.proj(x)
        h = self.posenc(h)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


def voicing_to_intervals(voicing_prob, t, threshold, min_dur_s=0.05):
    state = "off"
    intervals = []
    on_idx = -1
    for i, v in enumerate(voicing_prob):
        if state == "off" and v > threshold:
            state = "on"; on_idx = i
        elif state == "on" and v < threshold:
            state = "off"
            on_t, off_t = t[on_idx], t[i]
            if off_t - on_t >= min_dur_s:
                intervals.append([on_t, off_t])
    if state == "on":
        on_t, off_t = t[on_idx], t[-1]
        if off_t - on_t >= min_dur_s:
            intervals.append([on_t, off_t])
    return np.array(intervals)


def assign_pitch(intervals, t, pesto_hz):
    out = []
    for on, off in intervals:
        mask = (t >= on) & (t <= off) & (pesto_hz > 0)
        if mask.sum() > 0:
            out.append(float(np.median(pesto_hz[mask])))
        else:
            out.append(220.0)
    return np.array(out)


def score_clip(intervals_pred, pitches_pred, intervals_ref, pitches_ref):
    import mir_eval
    if len(intervals_pred) == 0 or len(intervals_ref) == 0:
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        intervals_ref, pitches_ref, intervals_pred, pitches_pred,
        offset_ratio=None, onset_tolerance=0.05,
    )[2]


def main(n_epochs: int = 60, d_model: int = 192, n_heads: int = 4,
         n_layers: int = 4, lr: float = 5e-4, threshold: float = 0.5) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "d_model": d_model,
             "n_heads": n_heads, "n_layers": n_layers, "lr": lr,
             "threshold": threshold, "arch": "transformer-encoder"}
    run = wandb.init(project="humscribe-v3.2", name="exp_B73_transformer",
                     config=cfg_w, tags=["B73", "vocadito", "transformer",
                                          "phase-d"],
                     dir="logs/wandb")

    print("loading cached features (extracted by B70)")
    qbh_data = []
    for wav in sorted(QBH.glob("*.wav")):
        try:
            f = load_or_extract(str(wav))
            iv, _ = pseudo_label_clip(str(wav))
            label = build_targets(f["t"], iv)
            qbh_data.append((wav.stem, f["feats"], label))
        except FileNotFoundError:
            continue
    voc_data = []
    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    for wav in sorted(audio_dir.glob("vocadito_*.wav")):
        a1 = ann_dir / f"{wav.stem}_notesA1.csv"
        if not a1.exists(): continue
        try:
            f = load_or_extract(str(wav))
        except FileNotFoundError:
            continue
        ref_iv, ref_pi = load_vocadito_notes(a1)
        label = build_targets(f["t"], ref_iv)
        voc_data.append((wav.stem, f["feats"], label, f["t"], f["pesto_hz"], ref_iv, ref_pi))
    print(f"  qbh {len(qbh_data)}, voc {len(voc_data)}, feat dim {voc_data[0][1].shape[1]}")

    np.random.seed(42)
    perm = np.random.permutation(len(voc_data))
    fold_size = len(voc_data) // 5
    folds = [perm[i*fold_size:(i+1)*fold_size].tolist() for i in range(5)]
    folds[-1].extend(perm[5*fold_size:].tolist())

    f1_per_fold = []
    in_dim = voc_data[0][1].shape[1]
    for fold_idx, val_idx in enumerate(folds):
        val_set = [voc_data[i] for i in val_idx]
        voc_train = [voc_data[i] for i in range(len(voc_data)) if i not in val_idx]
        combined = [(c, f_, lbl) for c, f_, lbl, *_ in voc_train] + qbh_data
        print(f"\n=== fold {fold_idx} ===")
        print(f"  train: {len(combined)}; val: {len(val_set)}")

        model = TransformerOnset(in_dim=in_dim, d_model=d_model, n_heads=n_heads,
                                   n_layers=n_layers).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        bce = nn.BCEWithLogitsLoss()
        best_f1 = 0.0

        for epoch in range(n_epochs):
            model.train()
            losses = []
            np.random.shuffle(combined)
            for c, feats, label in combined:
                feats_t = torch.from_numpy(feats[:3000]).unsqueeze(0).to("cuda")
                label_t = torch.from_numpy(label[:3000]).unsqueeze(0).to("cuda")
                opt.zero_grad()
                logits = model(feats_t)
                loss = bce(logits, label_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
            sched.step()
            mean_loss = sum(losses) / len(losses)
            wandb.log({"fold": fold_idx, "epoch": epoch, "train_loss": mean_loss,
                        "lr": sched.get_last_lr()[0]})

            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                model.eval()
                f1s = []
                with torch.no_grad():
                    for c, feats, label, t_axis, pesto_hz, ref_iv, ref_pi in val_set:
                        feats_t = torch.from_numpy(feats).unsqueeze(0).to("cuda")
                        logits = model(feats_t).squeeze(0).cpu().numpy()
                        voicing = 1 / (1 + np.exp(-logits))
                        pred_iv = voicing_to_intervals(voicing, t_axis, threshold)
                        if len(pred_iv) == 0:
                            f1s.append(0.0); continue
                        pred_pi = assign_pitch(pred_iv, t_axis, pesto_hz)
                        f1s.append(score_clip(pred_iv, pred_pi, ref_iv, ref_pi))
                mean_f1 = float(np.mean(f1s))
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                print(f"  fold {fold_idx} ep{epoch:3d} loss={mean_loss:.3f} val_F1={mean_f1:.4f} (best {best_f1:.4f})")
                wandb.log({"fold": fold_idx, "epoch": epoch, "val_f1": mean_f1,
                            "best_f1_so_far": best_f1})

        f1_per_fold.append(best_f1)

    overall = float(np.mean(f1_per_fold))
    print(f"\n5-fold CV mean (best_per_fold) F1: {overall:.4f}")
    wandb.summary["mean_cv_f1"] = overall
    wandb.summary["cv_std"] = float(np.std(f1_per_fold))
    OUT_JSON.write_text(json.dumps({"f1_per_fold": f1_per_fold, "mean_f1": overall,
                                     "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threshold", type=float, default=0.5)
    main(**vars(ap.parse_args()))
