"""ME-7 — anacrusis (pickup-note) detection.

Many beat-tracking misalignments come from `beat_this` putting "beat 1" at
the wrong note when the piece starts with a pickup (anacrusis): a short
upbeat note before the first downbeat. The standard fix is to detect a
short opening note that's lighter than what follows and shift the beat-1
hypothesis one note later.

Heuristic (Phase E spec): the first note is a pickup when:
- its duration < 0.6 * the mean duration of the next 4 notes, AND
- it lands within 300 ms of the first detected beat (suggesting it
  triggered beat_this to call that position beat 1), AND
- there are at least 4 subsequent notes (so we have stable context).

If the conditions match, return a beat-shift hint: the index of the first
"real" downbeat is 1 (skip note 0). The DP can apply this by subtracting
the pickup's value from beat positions or by shifting the beat grid.

Cheap. CPU-only. No training.
"""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humscribe.notes import NoteEvent


@dataclass
class AnacrusisResult:
    is_pickup: bool
    pickup_duration_s: float | None
    next_4_mean_duration_s: float | None
    reason: str


def detect_anacrusis(notes: Sequence[NoteEvent],
                     beats: np.ndarray | None = None,
                     pickup_dur_ratio: float = 0.6,
                     beat_proximity_s: float = 0.3,
                     min_context: int = 4) -> AnacrusisResult:
    """Return whether the first note looks like a pickup (anacrusis)."""
    if len(notes) < min_context + 1:
        return AnacrusisResult(False, None, None,
                                "fewer than min_context notes")
    first = notes[0]
    first_dur = first.offset_s - first.onset_s
    if first_dur <= 0:
        return AnacrusisResult(False, first_dur, None, "non-positive duration")
    next_durs = []
    for ev in notes[1: 1 + min_context]:
        d = ev.offset_s - ev.onset_s
        if d > 0:
            next_durs.append(d)
    if not next_durs:
        return AnacrusisResult(False, first_dur, None, "no valid context")
    mean_next = float(np.mean(next_durs))
    if first_dur >= pickup_dur_ratio * mean_next:
        return AnacrusisResult(False, first_dur, mean_next,
                                f"not short enough ({first_dur:.3f} vs "
                                f"{pickup_dur_ratio*mean_next:.3f} threshold)")
    # If beats are provided, check that the first note is close to beat[0].
    if beats is not None and len(beats) >= 1:
        if abs(first.onset_s - float(beats[0])) > beat_proximity_s:
            return AnacrusisResult(False, first_dur, mean_next,
                                    f"first note not near beat 0 "
                                    f"(dist={abs(first.onset_s-float(beats[0])):.3f}s)")
    return AnacrusisResult(True, first_dur, mean_next,
                            f"pickup detected (dur={first_dur:.3f} vs "
                            f"mean_next={mean_next:.3f})")


def shift_beats_for_pickup(beats: np.ndarray, downbeats: np.ndarray,
                            pickup_duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply a pickup-aware beat shift: move all beat times by +pickup_duration_s
    so the first downbeat lands on note index 1 instead of note 0."""
    shifted_beats = beats - pickup_duration_s
    shifted_db = downbeats - pickup_duration_s
    return shifted_beats, shifted_db
