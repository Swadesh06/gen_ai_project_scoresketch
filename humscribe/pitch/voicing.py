"""Convert frame-level (time, hz, voicing) into NoteEvents.

A pragmatic substitute for a full HMM segmenter: voicing-thresholded segments
are split whenever the smoothed MIDI semitone median changes by more than half
a semitone. See DESIGN_NOTES.md for why we picked this over Viterbi-on-pitch.
"""
from __future__ import annotations
import numpy as np

from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, hz_to_midi


def segment_pitch_to_notes(
    times: np.ndarray,
    hz: np.ndarray,
    voicing: np.ndarray,
    mc: ModeConfig,
) -> list[NoteEvent]:
    if len(times) == 0:
        return []
    midi = np.where(hz > 0, np.array([hz_to_midi(float(h)) for h in hz]), 0.0)
    smooth = _median_filter(midi, mc.pitch_smooth_window)
    voiced = voicing >= mc.voicing_threshold
    segs = _voiced_segments(times, voiced, mc.onset_merge_seconds)
    out: list[NoteEvent] = []
    for s_idx, e_idx in segs:
        sub_t = times[s_idx:e_idx + 1]
        sub_m = smooth[s_idx:e_idx + 1]
        sub_v = voicing[s_idx:e_idx + 1]
        out.extend(_split_on_pitch_change(sub_t, sub_m, sub_v, mc.min_note_seconds))
    return out


def _median_filter(x: np.ndarray, w: int) -> np.ndarray:
    w = max(int(w) | 1, 1)
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + w])
    return out


def _voiced_segments(
    times: np.ndarray, voiced: np.ndarray, merge_s: float,
) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    n = len(voiced)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and voiced[j + 1]:
            j += 1
        segs.append((i, j))
        i = j + 1
    if not segs:
        return segs
    merged: list[tuple[int, int]] = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if times[s] - times[pe] < merge_s:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _split_on_pitch_change(
    t: np.ndarray, m: np.ndarray, v: np.ndarray, min_s: float,
) -> list[NoteEvent]:
    if len(t) == 0:
        return []
    out: list[NoteEvent] = []
    start = 0
    cur_med = float(np.median(m[:max(int(len(m) * 0.2), 1)]))
    for k in range(1, len(t)):
        if abs(float(m[k]) - cur_med) > 0.5:
            note = _make_note(t, m, v, start, k - 1)
            if note.duration_s >= min_s:
                out.append(note)
            start = k
            cur_med = float(m[k])
    note = _make_note(t, m, v, start, len(t) - 1)
    if note.duration_s >= min_s:
        out.append(note)
    return out


def _make_note(t: np.ndarray, m: np.ndarray, v: np.ndarray, s: int, e: int) -> NoteEvent:
    midi_med = float(np.median(m[s:e + 1]))
    midi_int = int(round(midi_med)) if midi_med > 0 else 0
    hz = 440.0 * (2.0 ** ((midi_med - 69.0) / 12.0)) if midi_med > 0 else 0.0
    conf = float(np.mean(v[s:e + 1]))
    return NoteEvent(
        onset_s=float(t[s]),
        offset_s=float(t[e]) + (float(t[1] - t[0]) if len(t) > 1 else 0.01),
        pitch_hz=hz,
        pitch_midi=midi_int,
        confidence=conf,
    )
