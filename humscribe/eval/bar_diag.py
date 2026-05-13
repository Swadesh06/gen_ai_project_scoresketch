"""Phase G G-10: bar-level structural consistency diagnostic.

Reference-free score: 1 - normalised median-absolute-deviation of bar
durations. High score (close to 1) = bar boundaries are coherent;
low score (close to 0) = bar boundaries jitter (a structural failure
signal).

`bar_consistency(beats, downbeats)` is the public API. Returns a
score in [0, 1].
"""
from __future__ import annotations
import numpy as np


def bar_consistency(beats: np.ndarray, downbeats: np.ndarray) -> float:
    """1 - normalised MAD of consecutive downbeat-to-downbeat durations."""
    if downbeats is None or len(downbeats) < 3:
        return 0.0
    dbi = np.diff(np.asarray(downbeats, dtype=np.float64))
    dbi = dbi[(dbi > 0.05) & (dbi < 30.0)]
    if len(dbi) < 2:
        return 0.0
    med = float(np.median(dbi))
    if med <= 0:
        return 0.0
    mad = float(np.median(np.abs(dbi - med)))
    return float(max(0.0, 1.0 - (mad / med)))


def beat_consistency(beats: np.ndarray) -> float:
    """Same idea applied to consecutive beat IBIs."""
    if beats is None or len(beats) < 4:
        return 0.0
    ibis = np.diff(np.asarray(beats, dtype=np.float64))
    ibis = ibis[(ibis > 0.01) & (ibis < 5.0)]
    if len(ibis) < 2:
        return 0.0
    med = float(np.median(ibis))
    if med <= 0:
        return 0.0
    mad = float(np.median(np.abs(ibis - med)))
    return float(max(0.0, 1.0 - (mad / med)))
