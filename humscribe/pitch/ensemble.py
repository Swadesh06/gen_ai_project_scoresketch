"""Pitch-tracker ensembles.

- track_pitch_ensemble: per-frame max-confidence (B17, discarded — calibration
  mismatch made it lose to PESTO alone).
- track_pitch_hybrid_voicing: PESTO for pitch, CREPE periodicity for voicing
  (B36/B36b — wins by +5pp on Vocadito with vt=0.55-0.60). Slower (2x model
  inference) but a solid Vocadito improvement.
"""
from __future__ import annotations
import numpy as np

from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto


def track_pitch_ensemble(
    audio: np.ndarray,
    sr: int,
    crepe_voicing_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt, ph, pv = track_pitch_pesto(audio, sr)
    ct, ch, cv = track_pitch_crepe(audio, sr)
    if len(ct) == 0 or len(pt) == 0:
        return pt, ph, pv
    cv_scaled = np.clip(cv * crepe_voicing_scale, 0.0, 1.0)
    ch_at_p = np.interp(pt, ct, ch)
    cv_at_p = np.interp(pt, ct, cv_scaled)
    use_crepe = cv_at_p > pv
    out_hz = np.where(use_crepe, ch_at_p, ph)
    out_voicing = np.maximum(pv, cv_at_p)
    return pt, out_hz, out_voicing


def track_pitch_hybrid_voicing(
    audio: np.ndarray, sr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PESTO pitch + CREPE periodicity as voicing signal. (B36/B36b kept.)"""
    pt, ph, pv_pesto = track_pitch_pesto(audio, sr)
    ct, _ch, cv = track_pitch_crepe(audio, sr)
    if len(ct) == 0 or len(pt) == 0:
        return pt, ph, pv_pesto
    voicing = np.interp(pt, ct, cv)
    return pt, ph, voicing
