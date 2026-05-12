"""ME-1 — pYIN diversifier as an uncorrelated pitch vote.

pYIN (de Cheveigné & Kawahara 2002, with probabilistic extension by Mauch
& Dixon 2014) is a pure-DSP pitch tracker: no neural net, no training, no
GPU. It's shipped with librosa. Different failure modes from PESTO/CREPE:
fooled by noisy/breathy attacks rather than by harmonics (where neural
trackers can hallucinate).

This member returns a per-frame pitch estimate compatible with the
existing `humscribe.pitch` interface (times, hz, voicing). The intent is
that downstream code can ensemble pYIN with PESTO + CREPE-periodicity to
get an uncorrelated vote on disagreements.

Cheap. CPU-only.
"""
from __future__ import annotations
import numpy as np


def track_pitch_pyin(audio: np.ndarray, sr: int,
                     fmin: float = 50.0, fmax: float = 1000.0,
                     hop_length: int = 220,
                     frame_length: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """librosa.pyin; returns (times_s, hz, voicing).

    voicing is the librosa "voiced flag" cast to float (0/1).
    """
    import librosa
    f0, voiced, voiced_p = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr,
        hop_length=hop_length, frame_length=frame_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    # voiced_p is a probability (0..1) — keep it as the soft voicing signal.
    v = np.where(voiced, voiced_p, 0.0).astype(np.float32)
    return times.astype(np.float32), f0.astype(np.float32), v


def vote_with_pesto_crepe(t_pesto: np.ndarray, hz_pesto: np.ndarray, vc_pesto: np.ndarray,
                          t_pyin: np.ndarray, hz_pyin: np.ndarray, vc_pyin: np.ndarray,
                          agree_cents: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample pYIN onto PESTO's time base, return weighted average pitch + max voicing.

    When PESTO and pYIN agree (within `agree_cents`), confidence is boosted
    via max(vc_pesto, vc_pyin). When they disagree, PESTO wins (higher
    accuracy on average) and voicing is dampened.
    """
    # Resample pYIN to PESTO time grid.
    hz_p = np.interp(t_pesto, t_pyin, hz_pyin, left=0.0, right=0.0)
    vc_p = np.interp(t_pesto, t_pyin, vc_pyin, left=0.0, right=0.0)
    diff_cents = np.zeros_like(hz_pesto)
    has_both = (hz_pesto > 0) & (hz_p > 0)
    diff_cents[has_both] = 1200.0 * np.log2(
        np.maximum(hz_p[has_both], 1e-3) / np.maximum(hz_pesto[has_both], 1e-3)
    )
    agree = has_both & (np.abs(diff_cents) <= agree_cents)
    out_hz = hz_pesto.copy()  # always trust PESTO's pitch value
    out_vc = vc_pesto.copy()
    # Boost voicing on agreement, dampen on disagreement.
    out_vc[agree] = np.maximum(out_vc[agree], vc_p[agree])
    disagree = has_both & ~agree
    out_vc[disagree] = out_vc[disagree] * 0.7
    return t_pesto, out_hz.astype(np.float32), out_vc.astype(np.float32)
