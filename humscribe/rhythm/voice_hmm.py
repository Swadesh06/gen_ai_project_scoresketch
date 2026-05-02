"""HMM voice tracker (Phase B Exp B20).

Walks notes in onset-time order. State at step i = which voice (0..K-1) the
i-th note is assigned to. Hidden states are voices; observations are notes.

Transition cost: penalize voice switches based on gap-since-last-note in that
voice (long gap = legitimate switch; tiny gap = same-voice continuation
preferred). New-voice penalty when no existing voice is plausible.

Emission cost: gaussian on (pitch_diff_to_voice_last_pitch). A voice with no
prior note has uniform emission (-log K).

Beam search rather than full Viterbi: at most `beam_size` voices alive at any
time, the rest are "completed" voices that can no longer accept notes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import (
    default_allowed_durations, viterbi_quantize_rhythm,
)


@dataclass
class HMMVoiceConfig:
    pitch_sigma: float = 2.5         # semitones
    new_voice_cost: float = 6.0      # log-prob units; lower = more new voices
    same_voice_bonus: float = 1.5    # subtracted when assigning to same recent voice
    max_active_voices: int = 8       # beam width
    long_gap_s: float = 2.0          # gap > this = treat voice as completed
    keep_offset_min_dur_s: float = 0.05


def assign_voices_hmm(notes: Sequence[NoteEvent], cfg: HMMVoiceConfig | None = None) -> list[list[int]]:
    """Beam-search assignment. Returns list[voice_idx_list] in original index order."""
    cfg = cfg or HMMVoiceConfig()
    n = len(notes)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: notes[i].onset_s)
    voices: list[list[int]] = []
    voice_last_pitch: list[float] = []
    voice_last_offset: list[float] = []

    for k_, i in enumerate(order):
        n_i = notes[i]
        midi = n_i.midi()
        on = n_i.onset_s
        # close out voices with too-long gap so they don't compete
        # (they remain in `voices` for output but aren't candidates)
        active_idx = [v for v, lo in enumerate(voice_last_offset)
                      if (on - lo) <= cfg.long_gap_s]
        # cost of new voice
        best_cost = float(cfg.new_voice_cost)
        best_v = -1  # -1 means "create new voice"
        # cost for each active voice: gaussian on pitch
        for v in active_idx:
            lp = voice_last_pitch[v]
            d = midi - lp
            cost = 0.5 * (d / max(cfg.pitch_sigma, 1e-6)) ** 2
            # bonus for "same voice last touched recently"
            recency = on - voice_last_offset[v]
            if recency < 0.5:
                cost -= cfg.same_voice_bonus
            if cost < best_cost:
                best_cost = cost; best_v = v
        if best_v < 0:
            # cap voices
            if len(voices) >= cfg.max_active_voices and active_idx:
                # forced reassignment to least-bad active voice
                fallback = min(active_idx, key=lambda v: 0.5 * ((midi - voice_last_pitch[v]) / cfg.pitch_sigma) ** 2)
                voices[fallback].append(i)
                voice_last_pitch[fallback] = float(midi)
                voice_last_offset[fallback] = float(n_i.offset_s)
            else:
                voices.append([i])
                voice_last_pitch.append(float(midi))
                voice_last_offset.append(float(n_i.offset_s))
        else:
            voices[best_v].append(i)
            voice_last_pitch[best_v] = float(midi)
            voice_last_offset[best_v] = float(n_i.offset_s)
    return voices


def per_voice_durations(notes: Sequence[NoteEvent], voices: list[list[int]],
                        keep_offset_min_dur_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
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


def quantize_with_hmm_voice_tracking(
    notes: Sequence[NoteEvent],
    beats: np.ndarray,
    tatums_per_beat: int = 24,
    voice_cfg: HMMVoiceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    voices = assign_voices_hmm(notes, voice_cfg)
    onsets, adj_offsets = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(
        onsets, adj_offsets, beats, tatums_per_beat=tatums_per_beat,
        allowed_durations_tatums=default_allowed_durations(tatums_per_beat),
    )
