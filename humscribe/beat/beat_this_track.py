"""beat_this wrapper. Returns (beats, downbeats, bpm)."""
from __future__ import annotations
import numpy as np

from beat_this.inference import File2Beats


_CACHED: dict[str, "File2Beats"] = {}


def _get_model(checkpoint: str = "final0", device: str = "cpu") -> "File2Beats":
    key = f"{checkpoint}/{device}"
    if key not in _CACHED:
        _CACHED[key] = File2Beats(checkpoint_path=checkpoint, device=device)
    return _CACHED[key]


def track_beats_beat_this(
    audio_path: str,
    checkpoint: str = "final0",
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, float]:
    model = _get_model(checkpoint=checkpoint, device=device)
    beats, downbeats = model(audio_path)
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    bpm = _bpm_from_beats(beats)
    return beats, downbeats, bpm


def _bpm_from_beats(beats: np.ndarray) -> float:
    if len(beats) < 2:
        return 0.0
    iois = np.diff(beats)
    iois = iois[(iois > 0.2) & (iois < 2.0)]
    if len(iois) == 0:
        return 0.0
    return float(60.0 / np.median(iois))
