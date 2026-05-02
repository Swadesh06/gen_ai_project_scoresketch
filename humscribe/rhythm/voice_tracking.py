"""Voice tracking + per-voice rhythm quantization (Phase B Exp B15).

Greedy assignment: walk ByteDance notes in onset-time order, attach each note
to the most recent voice whose last note is (a) within `time_gap_s` of the new
note's onset and (b) within `pitch_jump` semitones of the new note's pitch.
Otherwise start a new voice.

Once notes are partitioned by voice, durations are estimated as
`next_onset_in_same_voice - current_onset` (capped by the original offset).
This converts polyphonic-overlap noise into monophonic-clean durations
per-voice, which the Cemgil-Kappen DP handles well.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import (
    default_allowed_durations, viterbi_quantize_rhythm,
)


@dataclass
class VoiceTrackConfig:
    pitch_jump: float = 4.0          # semitones; soprano usually moves <= 4 per step
    time_gap_s: float = 1.5          # voice times out after this gap
    keep_offset_min_dur_s: float = 0.05


def assign_voices(notes: Sequence[NoteEvent], cfg: VoiceTrackConfig | None = None) -> list[list[int]]:
    """Return list of voices, each a list of indices into `notes`."""
    cfg = cfg or VoiceTrackConfig()
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset_s)
    voices: list[list[int]] = []
    voice_last_pitch: list[int] = []
    voice_last_offset: list[float] = []
    for i in order:
        n = notes[i]
        midi = n.midi()
        best_v = -1; best_d = float("inf")
        for v_idx, (lp, lo) in enumerate(zip(voice_last_pitch, voice_last_offset)):
            if (n.onset_s - lo) > cfg.time_gap_s:
                continue
            d = abs(midi - lp)
            if d <= cfg.pitch_jump and d < best_d:
                best_d = d; best_v = v_idx
        if best_v < 0:
            voices.append([i]); voice_last_pitch.append(midi); voice_last_offset.append(n.offset_s)
        else:
            voices[best_v].append(i)
            voice_last_pitch[best_v] = midi
            voice_last_offset[best_v] = n.offset_s
    return voices


def per_voice_durations(notes: Sequence[NoteEvent], voices: list[list[int]],
                        keep_offset_min_dur_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """For each note, its duration = min(original_dur, gap_to_next_in_voice).

    Returns onsets and adjusted offsets in seconds, in original index order."""
    n = len(notes)
    onsets = np.array([notes[i].onset_s for i in range(n)], dtype=np.float64)
    offsets = np.array([notes[i].offset_s for i in range(n)], dtype=np.float64)
    adjusted = offsets.copy()
    for v in voices:
        if len(v) <= 1:
            continue
        v_sorted = sorted(v, key=lambda i: notes[i].onset_s)
        for k in range(len(v_sorted) - 1):
            i = v_sorted[k]; j = v_sorted[k + 1]
            next_on = float(notes[j].onset_s)
            dur = next_on - float(notes[i].onset_s)
            if keep_offset_min_dur_s <= dur < adjusted[i] - float(notes[i].onset_s):
                adjusted[i] = float(notes[i].onset_s) + dur
    return onsets, adjusted


def quantize_with_voice_tracking(
    notes: Sequence[NoteEvent],
    beats: np.ndarray,
    tatums_per_beat: int = 24,
    voice_cfg: VoiceTrackConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    voices = assign_voices(notes, voice_cfg)
    onsets, adj_offsets = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(
        onsets, adj_offsets, beats, tatums_per_beat=tatums_per_beat,
        allowed_durations_tatums=default_allowed_durations(tatums_per_beat),
    )
