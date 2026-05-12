"""Beat-tempo octave sanity check (Phase F-1 rule-based corrector).

beat_this's `target_bpm` parameter biases the picked tempo octave toward
a center value. But for slow pieces like Chopin Berceuse (~30 BPM) the
default target_bpm=110 puts the closest log2-octave at 120 BPM — wrong
by 2× or 4×. This module detects octave misalignment using the predicted
notes' density and provides a correction.

Detection signal:
- Compute median IOI of the predicted notes.
- Compute predicted beat IOI.
- If notes/beat ratio is very low (< 0.5) or very high (> 8), the
  predicted beat octave is mismatched against the actual music.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence

from humscribe.notes import NoteEvent


def _median_ioi(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.5
    iois = np.diff(np.sort(times))
    iois = iois[(iois > 0.01) & (iois < 10.0)]
    if len(iois) == 0:
        return 0.5
    return float(np.median(iois))


def detect_octave_misalignment(beats: np.ndarray,
                                notes: Sequence[NoteEvent]) -> dict:
    """Return diagnostic dict. The 'recommend' field is one of:
      - 'keep' (no action)
      - 'halve' (beat_this detected 2× the real tempo; remove every other beat)
      - 'double' (beat_this detected 0.5× the real tempo; insert half-beats)
    """
    if len(beats) < 2 or len(notes) == 0:
        return {"recommend": "keep", "reason": "insufficient data"}
    beat_iois = np.diff(beats)
    beat_iois = beat_iois[(beat_iois > 0.01) & (beat_iois < 5.0)]
    if len(beat_iois) == 0:
        return {"recommend": "keep", "reason": "no usable beat ibis"}
    median_bpb_ioi = float(np.median(beat_iois))
    note_times = np.array([n.onset_s for n in notes])
    median_note_ioi = _median_ioi(note_times)
    # notes_per_beat = beat_ioi / note_ioi. Typical piano: 2-6 notes per beat.
    notes_per_beat = median_bpb_ioi / max(median_note_ioi, 1e-3)
    # If notes_per_beat > 8: beats are too sparse — each predicted beat
    # covers many notes; the real tempo is faster, so DOUBLE the beats.
    # If notes_per_beat < 0.4: beats are too dense — many beats per note;
    # the real tempo is slower, so HALVE the beats.
    # Empirically tuned on the 9 ASAP pieces.
    # Halve signal: a "fast-tempo" detection (BPM > 100) combined with
    # slow-note signal (note_ioi > 0.4 s) is contradictory — fast tempo
    # but each note > 1 beat. That means the detected beats are subdivisions
    # of the true beats. Catches Chopin Berceuse (BPM=120, note_ioi=0.5).
    pred_bpm = 60.0 / max(median_bpb_ioi, 1e-3)
    fast_tempo_slow_note = (pred_bpm > 100.0 and median_note_ioi > 0.35
                              and notes_per_beat < 1.5)
    if notes_per_beat > 5.5:
        rec = "double"
    elif fast_tempo_slow_note or notes_per_beat < 0.4:
        rec = "halve"
    else:
        rec = "keep"
    return {"recommend": rec,
            "median_beat_ioi": median_bpb_ioi,
            "median_note_ioi": median_note_ioi,
            "notes_per_beat": notes_per_beat,
            "n_beats": int(len(beats)),
            "n_notes": int(len(notes))}


def apply_octave_correction(beats: np.ndarray, downbeats: np.ndarray,
                             recommend: str) -> tuple[np.ndarray, np.ndarray]:
    if recommend == "halve":
        # Keep every other beat. First beat is preserved.
        return beats[::2], downbeats
    if recommend == "double":
        # Insert midpoints. Downbeats stay.
        new_beats = []
        for i in range(len(beats) - 1):
            mid = 0.5 * (beats[i] + beats[i + 1])
            new_beats.append(beats[i])
            new_beats.append(mid)
        new_beats.append(beats[-1])
        return np.array(new_beats, dtype=np.float64), downbeats
    return beats, downbeats
