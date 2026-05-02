"""Per-frame max-confidence ensemble of PESTO and CREPE pitch tracks (Phase B Exp B17)."""
from __future__ import annotations
import numpy as np

from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto


def track_pitch_ensemble(
    audio: np.ndarray,
    sr: int,
    crepe_voicing_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run both PESTO and CREPE; align to PESTO's frame grid; pick per-frame
    pitch from the more-confident tracker (after rescaling CREPE's periodicity
    to PESTO's confidence scale)."""
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
