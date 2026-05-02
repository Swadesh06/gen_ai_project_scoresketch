"""beat_this wrapper. Returns (beats, downbeats, bpm).

Optionally accepts a `target_bpm` for tempo-octave correction (Phase B Exp B13):
if the predicted BPM is half or double the target, re-interpolate beats to the
target tempo octave. This fixes beat_this's common confusion between
quarter-note and half-note pulses.
"""
from __future__ import annotations
import numpy as np
import torch

from beat_this.inference import File2Beats


_CACHED: dict[str, "File2Beats"] = {}


def _autodevice() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(checkpoint: str = "final0", device: str | None = None) -> "File2Beats":
    dev = device or _autodevice()
    key = f"{checkpoint}/{dev}"
    if key not in _CACHED:
        _CACHED[key] = File2Beats(checkpoint_path=checkpoint, device=dev)
    return _CACHED[key]


def track_beats_beat_this(
    audio_path: str,
    checkpoint: str = "final0",
    device: str | None = None,
    target_bpm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = _get_model(checkpoint=checkpoint, device=device)
    beats, downbeats = model(audio_path)
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    bpm = _bpm_from_beats(beats)
    if target_bpm and bpm > 0:
        beats, downbeats, bpm = _maybe_octave_correct(beats, downbeats, bpm, target_bpm)
    return beats, downbeats, bpm


def _bpm_from_beats(beats: np.ndarray) -> float:
    if len(beats) < 2:
        return 0.0
    iois = np.diff(beats)
    iois = iois[(iois > 0.05) & (iois < 5.0)]
    if len(iois) == 0:
        return 0.0
    return float(60.0 / np.median(iois))


def _maybe_octave_correct(beats: np.ndarray, downbeats: np.ndarray,
                          bpm: float, target_bpm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Snap to the nearest tempo octave of the predicted bpm to target_bpm.
    Doubles by inserting midpoints; halves by dropping every other beat."""
    options = [(bpm, beats)]
    if len(beats) >= 2:
        midpoints = (beats[:-1] + beats[1:]) / 2.0
        doubled = np.sort(np.concatenate([beats, midpoints]))
        options.append((bpm * 2.0, doubled))
    if len(beats) >= 4:
        halved = beats[::2]
        options.append((bpm / 2.0, halved))
    diffs = [abs(np.log2(max(b, 1.0) / target_bpm)) for b, _ in options]
    j = int(np.argmin(diffs))
    new_bpm, new_beats = options[j]
    return new_beats, downbeats, float(new_bpm)
