"""PESTO pitch tracker. Returns (times_s, hz, voicing_confidence)."""
from __future__ import annotations
import numpy as np
import torch

import pesto


def track_pitch_pesto(
    audio: np.ndarray,
    sr: int,
    step_size_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    arr = audio.astype(np.float32, copy=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(arr).to(device)
    timesteps, pitch_hz, conf, _act = pesto.predict(
        x, sr, step_size=float(step_size_ms),
    )
    times = timesteps.detach().cpu().numpy().astype(np.float64) / 1000.0
    hz = pitch_hz.detach().cpu().numpy().astype(np.float64)
    voicing = conf.detach().cpu().numpy().astype(np.float64)
    return times, hz, voicing
