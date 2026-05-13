"""Phase G G-9: confidence-aware per-note output.

Every component in the pipeline (PESTO pitch, CREPE periodicity, beat_this
strength, ByteDance velocity, YourMT3+ token logits) emits a confidence
in [0, 1]. We aggregate them into a single per-note confidence and
forward it to `NoteEvent.confidence`.

Aggregation:
- humming: mean(pesto_conf) * mean(crepe_periodicity) * beat_strength_at_onset
- piano/instrument: mean(transcriber_conf) * beat_strength_at_onset

Two public helpers:
- `aggregate_confidence(notes, pesto_trace, crepe_trace, beats)` mutates
  `NoteEvent.confidence` in-place for the humming branch.
- `global_confidence(notes)` returns a per-piece scalar (mean note conf).
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from humscribe.notes import NoteEvent


def _frame_mean(arr: np.ndarray, t: np.ndarray, on: float, off: float) -> float:
    if len(arr) == 0 or len(t) == 0:
        return 1.0
    lo = int(np.searchsorted(t, on, side="left"))
    hi = int(np.searchsorted(t, off, side="right"))
    if hi <= lo:
        return float(arr[min(lo, len(arr) - 1)])
    seg = arr[lo:hi]
    if seg.size == 0:
        return 1.0
    return float(np.clip(np.mean(seg), 0.0, 1.0))


def _beat_strength_at(onset_s: float, beats: np.ndarray) -> float:
    if beats is None or len(beats) < 2:
        return 1.0
    ibis = np.diff(beats)
    if len(ibis) == 0:
        return 1.0
    med_ibi = float(np.median(ibis))
    if med_ibi <= 0:
        return 1.0
    nearest_beat = float(np.min(np.abs(beats - onset_s)))
    return float(np.clip(1.0 - (nearest_beat / max(med_ibi, 1e-3)), 0.0, 1.0))


def aggregate_confidence(notes: Sequence[NoteEvent],
                         pesto_trace: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                         crepe_trace: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                         beats: np.ndarray | None) -> None:
    """In-place: set `NoteEvent.confidence` to the aggregated score.

    pesto_trace / crepe_trace = (times, hz, conf) tuples.
    Pass None for the missing branch (e.g. instrument).
    """
    t_p, _hzp, vp = pesto_trace if pesto_trace is not None else (np.zeros(0),) * 3
    t_c, _hzc, vc = crepe_trace if crepe_trace is not None else (np.zeros(0),) * 3
    for n in notes:
        p_c = _frame_mean(vp, t_p, n.onset_s, n.offset_s)
        c_c = _frame_mean(vc, t_c, n.onset_s, n.offset_s)
        b_c = _beat_strength_at(n.onset_s, beats) if beats is not None else 1.0
        n.confidence = float(np.clip(p_c * c_c * b_c, 0.0, 1.0))


def global_confidence(notes: Sequence[NoteEvent]) -> float:
    if not notes:
        return 0.0
    return float(np.mean([n.confidence for n in notes]))
