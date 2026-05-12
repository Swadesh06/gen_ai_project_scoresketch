"""ME-10 — meter-template ensemble.

The time-signature heuristic in `humscribe.pipeline._infer_time_signature`
infers a single time signature from `beat_this`'s downbeat spacing. When
the downbeat detection is unreliable (humming, dense Romantic piano)
the inferred time sig is wrong and downstream rendering of the music21
score has mis-placed bar lines and unreadable measure groupings.

ME-10's approach: run the rhythm-DP under N candidate time-signature
hypotheses {2/4, 3/4, 4/4, 6/8, 12/8} independently, score the resulting
note grouping by how well it fits a metric template (strong beats heavy,
weak beats light), and pick the hypothesis with the best alignment.

Embarrassingly parallel — each hypothesis can run on its own CPU.
"""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humscribe.notes import NoteEvent


CANDIDATES: list[tuple[int, int]] = [
    (2, 4), (3, 4), (4, 4), (6, 8), (12, 8),
]

# Standard metric-strength template per beat position. Higher = stronger beat.
# Values are arbitrary but follow the textbook strong/weak pattern.
TEMPLATES: dict[tuple[int, int], list[float]] = {
    (2, 4): [3.0, 1.0],  # strong-weak
    (3, 4): [3.0, 1.0, 1.0],  # strong-weak-weak
    (4, 4): [3.0, 1.0, 2.0, 1.0],  # strong-weak-medium-weak
    (6, 8): [3.0, 1.0, 1.0, 2.0, 1.0, 1.0],
    (12, 8): [3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
}


@dataclass
class MeterScore:
    time_sig: tuple[int, int]
    score: float
    n_strong_aligned: int
    n_notes: int


def _strong_alignment_score(notes: Sequence[NoteEvent],
                             beats: np.ndarray,
                             time_sig: tuple[int, int]) -> MeterScore:
    """Score how well notes' onsets align with strong beats of `time_sig`."""
    num, _ = time_sig
    template = TEMPLATES.get(time_sig, [1.0] * num)
    if len(beats) < 2 or not notes:
        return MeterScore(time_sig, 0.0, 0, len(notes))
    # For each note onset, compute its beat index (mod num) by matching to
    # the nearest beat time.
    on_times = np.array([n.onset_s for n in notes])
    beat_for_note = np.searchsorted(beats, on_times, side="right") - 1
    beat_for_note = np.clip(beat_for_note, 0, len(beats) - 1)
    # Position within bar = beat index mod numerator.
    pos_in_bar = beat_for_note % num
    weights = np.array([template[p] for p in pos_in_bar])
    # Score = total weight (notes on strong beats contribute more).
    score = float(weights.sum())
    n_strong = int(((np.array(template) >= 2.0)[pos_in_bar]).sum())
    return MeterScore(time_sig, score, n_strong, len(notes))


def best_time_signature(notes: Sequence[NoteEvent],
                         beats: np.ndarray) -> MeterScore:
    """Pick the best time signature hypothesis by metric-template fit."""
    if len(beats) < 2 or not notes:
        return MeterScore((4, 4), 0.0, 0, len(notes))
    scores = [_strong_alignment_score(notes, beats, ts) for ts in CANDIDATES]
    # Normalise per-note so longer-numerator hypotheses don't auto-win.
    best = max(scores, key=lambda s: s.score / max(s.n_notes, 1))
    return best
