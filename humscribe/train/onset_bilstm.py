"""Tiny BiLSTM onset detector (Phase B Exp B10).

Inputs per frame: (midi_obs_normalized, voicing, log-energy_proxy_via_voicing).
Output: per-frame onset probability.

Trained on Vocadito, validated on held-out clips. Replaces the voicing-based
onset trigger in the segmenter when used.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn

from humscribe.notes import NoteEvent, hz_to_midi, midi_to_hz


@dataclass
class OnsetModelConfig:
    in_dim: int = 3
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2


class OnsetBiLSTM(nn.Module):
    def __init__(self, cfg: OnsetModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            cfg.in_dim, cfg.hidden, num_layers=cfg.layers,
            bidirectional=True, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.hidden, cfg.hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def make_features(hz: np.ndarray, voicing: np.ndarray) -> np.ndarray:
    midi = np.where(hz > 0, np.array([hz_to_midi(float(h)) for h in hz]), 0.0)
    midi_n = (midi - 60.0) / 24.0
    energy_proxy = (voicing ** 2)
    return np.stack([midi_n.astype(np.float32), voicing.astype(np.float32),
                     energy_proxy.astype(np.float32)], axis=-1)


def make_labels(onset_times_s: np.ndarray, frame_times: np.ndarray, hop: float = 0.01) -> np.ndarray:
    y = np.zeros(len(frame_times), dtype=np.float32)
    if len(onset_times_s) == 0:
        return y
    for ot in onset_times_s:
        idx = int(round(float(ot) / hop))
        for d in (-1, 0, 1):
            if 0 <= idx + d < len(y):
                y[idx + d] = 1.0
    return y


def predict_onsets(model: OnsetBiLSTM, hz: np.ndarray, voicing: np.ndarray,
                   threshold: float = 0.5, device: str = "cpu") -> np.ndarray:
    model.eval()
    x = torch.from_numpy(make_features(hz, voicing)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    is_onset = probs > threshold
    out = np.zeros_like(is_onset, dtype=bool)
    for i in range(len(is_onset)):
        if is_onset[i] and (i == 0 or not out[max(0, i - 5):i].any()):
            out[i] = True
    return out


def segment_via_onsets(times: np.ndarray, hz: np.ndarray, voicing: np.ndarray,
                       onset_mask: np.ndarray, voicing_threshold: float = 0.30,
                       min_note_seconds: float = 0.05) -> list[NoteEvent]:
    if len(times) == 0:
        return []
    starts = np.where(onset_mask)[0]
    if len(starts) == 0:
        return []
    notes: list[NoteEvent] = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 1 if k + 1 < len(starts) else len(times) - 1
        slice_v = voicing[s:e + 1]
        if slice_v.mean() < voicing_threshold * 0.5:
            continue
        slice_h = hz[s:e + 1]
        slice_t = times[s:e + 1]
        valid = slice_h > 0
        if not valid.any():
            continue
        midi_med = float(np.median([hz_to_midi(float(h)) for h in slice_h[valid]]))
        midi_int = int(round(midi_med)) if midi_med > 0 else 0
        on_t = float(slice_t[0])
        off_t = float(slice_t[-1]) + (float(times[1] - times[0]) if len(times) > 1 else 0.01)
        if (off_t - on_t) < min_note_seconds:
            continue
        notes.append(NoteEvent(
            onset_s=on_t, offset_s=off_t,
            pitch_hz=midi_to_hz(midi_med), pitch_midi=midi_int,
            confidence=float(slice_v.mean()),
        ))
    return notes


def train_loop(features: list[np.ndarray], labels: list[np.ndarray],
               cfg: OnsetModelConfig | None = None, epochs: int = 60,
               batch_size: int = 4, lr: float = 1e-3, device: str = "cuda",
               val_features: list[np.ndarray] | None = None,
               val_labels: list[np.ndarray] | None = None) -> tuple[OnsetBiLSTM, list[dict]]:
    cfg = cfg or OnsetModelConfig()
    model = OnsetBiLSTM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = _pos_weight(labels).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history: list[dict] = []
    n = len(features)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(n)
        model.train()
        train_loss = 0.0
        for i in range(0, n, batch_size):
            ids = order[i:i + batch_size]
            xs = [torch.from_numpy(features[j]) for j in ids]
            ys = [torch.from_numpy(labels[j]) for j in ids]
            xb, yb, mask = _pad_batch(xs, ys, device)
            logits = model(xb)
            loss = (bce(logits, yb) * mask).sum() / mask.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.detach()) * len(ids)
        train_loss /= n
        rec = {"epoch": ep, "train_loss": train_loss}
        if val_features is not None and val_labels is not None:
            model.eval()
            vl, vp, vr, vf = _val_metrics(model, val_features, val_labels, device)
            rec.update({"val_loss": vl, "val_p": vp, "val_r": vr, "val_f1": vf})
        history.append(rec)
    return model, history


def _pos_weight(labels: list[np.ndarray]) -> torch.Tensor:
    n_pos = sum(int(l.sum()) for l in labels)
    n_neg = sum(int((1 - l).sum()) for l in labels)
    return torch.tensor(max(n_neg / max(n_pos, 1), 1.0), dtype=torch.float32)


def _pad_batch(xs, ys, device):
    T = max(x.shape[0] for x in xs)
    D = xs[0].shape[1]
    xb = torch.zeros(len(xs), T, D)
    yb = torch.zeros(len(xs), T)
    mask = torch.zeros(len(xs), T)
    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        xb[i, :L] = x
        yb[i, :L] = y
        mask[i, :L] = 1.0
    return xb.to(device), yb.to(device), mask.to(device)


def _val_metrics(model, val_features, val_labels, device):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0; total_n = 0
    tp = fp = fn = 0
    for f, l in zip(val_features, val_labels):
        x = torch.from_numpy(f).unsqueeze(0).to(device)
        y = torch.from_numpy(l).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            ll = bce(logits, y).mean()
            total_loss += float(ll) * f.shape[0]
            total_n += f.shape[0]
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]
        gt = l > 0.5
        tp += int(np.sum(pred & gt))
        fp += int(np.sum(pred & ~gt))
        fn += int(np.sum(~pred & gt))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return total_loss / max(total_n, 1), float(p), float(r), float(f1)
