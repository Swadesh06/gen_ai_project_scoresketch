"""CREPE pitch tracker (via torchcrepe). Returns (times_s, hz, voicing)."""
from __future__ import annotations
import numpy as np
import torch
import torchcrepe


def track_pitch_crepe(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 10.0,
    model: str = "full",
    fmin: float = 50.0,
    fmax: float = 1100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    arr = audio.astype(np.float32, copy=False)
    x = torch.from_numpy(arr).unsqueeze(0)
    target_sr = 16000
    if sr != target_sr:
        x = torchcrepe.resample(x, sr, target_sr)
        sr = target_sr
    hop = max(int(round(hop_ms * 1e-3 * sr)), 1)
    pitch, periodicity = torchcrepe.predict(
        x, sr, hop_length=hop, fmin=fmin, fmax=fmax, model=model,
        return_periodicity=True, batch_size=512, device="cpu",
    )
    hz = pitch.squeeze(0).detach().cpu().numpy().astype(np.float64)
    voicing = periodicity.squeeze(0).detach().cpu().numpy().astype(np.float64)
    times = np.arange(len(hz), dtype=np.float64) * (hop / sr)
    return times, hz, voicing
