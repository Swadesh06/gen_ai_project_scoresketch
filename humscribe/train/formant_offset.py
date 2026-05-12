"""Phase F-2 — formant-band offset detector training.

Tiny BiLSTM that consumes the cached formant mel-spectrogram (80-bin,
1500-3500 Hz, 10 ms hop) and predicts per-frame "note ends here" logit.
Targets the Vocadito offset20 gap (current 0.439 vs IAA 0.642).

Uses Vocadito A1 annotations as labels. 5-fold CV.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class FormantOffsetConfig:
    in_dim: int = 80
    hidden: int = 96
    layers: int = 2
    dropout: float = 0.2


class FormantOffsetBiLSTM(nn.Module):
    def __init__(self, cfg: FormantOffsetConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.in_dim, cfg.hidden, num_layers=cfg.layers,
                             bidirectional=True, batch_first=True,
                             dropout=cfg.dropout if cfg.layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(2 * cfg.hidden, cfg.hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def make_offset_labels(offset_times_s: np.ndarray, n_frames: int,
                        hop_s: float = 0.01) -> np.ndarray:
    """1.0 within ±1 frame of an offset event."""
    y = np.zeros(n_frames, dtype=np.float32)
    for ot in offset_times_s:
        idx = int(round(float(ot) / hop_s))
        for d in (-1, 0, 1):
            if 0 <= idx + d < n_frames:
                y[idx + d] = 1.0
    return y
