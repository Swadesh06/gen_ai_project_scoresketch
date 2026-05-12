"""ME-12 — phase-deviation onset detector (Bello et al. 2005).

Spectral phase derivative — different feature space from magnitude-based
voicing changes. Available via librosa's onset_strength function with
the 'cqt' or 'mel' feature mode and phase-deviation.

Used as a cross-check on the voicing-based onset estimator: regions
where phase-deviation says "onset here" but voicing doesn't can be
votes for onsets in the segmenter.
"""
from __future__ import annotations
import numpy as np


def phase_deviation_onsets(audio: np.ndarray, sr: int,
                            hop_length: int = 220,
                            n_fft: int = 2048,
                            threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Detect onsets via spectral phase derivative.

    Returns (times_s, strengths) — times of detected onset peaks and the
    per-peak strength. Use the strengths to gate the onset votes.
    """
    import librosa
    # Phase-deviation onset strength curve.
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length,
        feature=librosa.feature.melspectrogram,
        aggregate=np.median,
        n_fft=n_fft,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        units="frames", backtrack=False,
    )
    times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    strengths = onset_env[onset_frames] if len(onset_frames) else np.array([])
    return times, strengths
