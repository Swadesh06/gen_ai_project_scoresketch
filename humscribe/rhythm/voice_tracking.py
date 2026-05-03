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
    # Tuned by exp_B16 sweep (BWV 846: snap 0.779 -> 0.847 with these defaults).
    pitch_jump: float = 3.0          # semitones; tight to keep voice lines coherent
    time_gap_s: float = 0.5          # short timeout: aggressive new-voice creation
    keep_offset_min_dur_s: float = 0.05


def adaptive_pitch_jump(notes: Sequence[NoteEvent]) -> float:
    """Auto-select pitch_jump based on note pitch-spread per second (B48b finding).
    Bach Fugues (4 voices, narrow ranges) -> tight pj=3.
    Romantic (wide chordal textures) -> wider pj=12.
    """
    if len(notes) < 10:
        return 3.0
    span = max(notes[-1].onset_s - notes[0].onset_s, 1e-3)
    notes_per_sec = len(notes) / span
    midis = [n.midi() for n in notes if n.midi() > 0]
    if not midis:
        return 3.0
    pitch_iqr = float(np.percentile(midis, 75) - np.percentile(midis, 25))
    if notes_per_sec > 6.0 and pitch_iqr > 24:
        return 12.0
    if notes_per_sec > 4.0 and pitch_iqr > 18:
        return 7.0
    return 3.0


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
    adaptive_pj: bool = True,
    per_voice_dp: bool = False,
    voice_assigner=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Voice-track + Cemgil DP, optionally one DP per voice (B79).

    Args:
        adaptive_pj: auto-select pitch_jump (B49). Default True.
        per_voice_dp: if True, run viterbi_quantize_rhythm INDEPENDENTLY on each
            voice and merge — wins +1.7pp on Chopin Berceuse-style pieces with
            clear melody+accompaniment separation (B79). Default False keeps
            production behavior (single shared DP on adjusted offsets).
        voice_assigner: optional callable `(notes) -> list[list[int]]` to use
            instead of the greedy assigner. Used by B78 / future learned voice
            trackers (B76 Transformer).
    """
    if not notes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if voice_cfg is None:
        voice_cfg = VoiceTrackConfig()
    if adaptive_pj:
        voice_cfg = VoiceTrackConfig(
            pitch_jump=adaptive_pitch_jump(notes),
            time_gap_s=voice_cfg.time_gap_s,
            keep_offset_min_dur_s=voice_cfg.keep_offset_min_dur_s,
        )
    voices = (voice_assigner(notes) if voice_assigner is not None
               else assign_voices(notes, voice_cfg))
    if per_voice_dp:
        return _per_voice_dp(notes, voices, beats, tatums_per_beat)
    onsets, adj_offsets = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(
        onsets, adj_offsets, beats, tatums_per_beat=tatums_per_beat,
        allowed_durations_tatums=default_allowed_durations(tatums_per_beat),
    )


def _per_voice_dp(notes, voices, beats, tatums_per_beat):
    """Run viterbi_quantize_rhythm independently per voice, merge to global order.

    Voice members are quantized in beat-relative time using ALL beats (not just
    those spanning the voice's notes), so cross-voice tatum positions remain
    comparable. B79 finding: this wins ~+1.7pp on melody+accompaniment pieces."""
    n = len(notes)
    q_on = np.zeros(n, dtype=np.int64)
    q_off = np.zeros(n, dtype=np.int64)
    allowed = default_allowed_durations(tatums_per_beat)
    for v in voices:
        if not v:
            continue
        v_sorted = sorted(v, key=lambda i: notes[i].onset_s)
        v_on = np.array([notes[i].onset_s for i in v_sorted], dtype=np.float64)
        v_off = np.array([notes[i].offset_s for i in v_sorted], dtype=np.float64)
        # Cap each note's offset at gap-to-next-in-voice (same as per_voice_durations)
        v_off_adj = v_off.copy()
        for k in range(len(v_sorted) - 1):
            gap = v_on[k + 1] - v_on[k]
            if 0.05 <= gap < v_off_adj[k] - v_on[k]:
                v_off_adj[k] = v_on[k] + gap
        q_on_v, q_off_v = viterbi_quantize_rhythm(
            v_on, v_off_adj, beats, tatums_per_beat=tatums_per_beat,
            allowed_durations_tatums=allowed,
        )
        for ki, gi in enumerate(v_sorted):
            q_on[gi] = q_on_v[ki]
            q_off[gi] = q_off_v[ki]
    return q_on, q_off
