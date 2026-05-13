"""Phase G G-4 + G-5 + G-6: published note-event post-processing tricks.

These run on the already-segmented `list[NoteEvent]` (humming branch) or
on the audio prior to beat tracking. They're separate from segmentation
so they can be A/B-tested cleanly.

G-4: same-pitch gap merging (CREPE Notes 2023). Two consecutive notes of
identical MIDI pitch separated by < 80 ms are vibrato fragments of a
single held note. Merge them.

G-5: median pitch smoothing (Mauch & Dixon 2014 pYIN). 250 ms moving
median on voiced frames preserves unvoiced markers. This is a wider
window than the segmenter's default 190 ms (19 frames at 10 ms hop).

G-6: silent-region trimming. Strip leading/trailing silence below
-40 dB FS so beat_this doesn't place beats in silence. Margin 10 ms.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from humscribe.notes import NoteEvent


def merge_same_pitch(notes: Sequence[NoteEvent], gap_s: float = 0.080) -> list[NoteEvent]:
    """G-4: merge consecutive same-pitch NoteEvents within `gap_s`.

    Preserves the earliest onset and the latest offset across the merge.
    Confidence becomes the duration-weighted mean. Notes must be in onset
    order; if they aren't, this sorts them.
    """
    if not notes:
        return []
    items = sorted(notes, key=lambda n: n.onset_s)
    out: list[NoteEvent] = [items[0]]
    for n in items[1:]:
        prev = out[-1]
        if (n.pitch_midi == prev.pitch_midi and n.pitch_midi is not None
                and (n.onset_s - prev.offset_s) <= gap_s):
            new_offset = max(prev.offset_s, n.offset_s)
            d_prev = max(prev.offset_s - prev.onset_s, 1e-6)
            d_cur = max(n.offset_s - n.onset_s, 1e-6)
            conf = (prev.confidence * d_prev + n.confidence * d_cur) / (d_prev + d_cur)
            out[-1] = NoteEvent(
                onset_s=prev.onset_s, offset_s=new_offset,
                pitch_hz=prev.pitch_hz, pitch_midi=prev.pitch_midi,
                velocity=prev.velocity, confidence=conf,
            )
        else:
            out.append(n)
    return out


def median_smooth_pitch(times: np.ndarray, hz: np.ndarray, voicing: np.ndarray,
                         window_ms: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    """G-5: 250-ms moving-median smoothing on the voiced portion of `hz`.

    Computes the frame hop from `times` and rounds the window to an odd
    number of frames. Unvoiced frames (voicing below 0.05) keep their
    original hz (typically 0) so segmentation can still pick up
    voiced/unvoiced transitions.
    """
    n = len(times)
    if n < 2:
        return hz.copy(), voicing.copy()
    hop_s = max(float(times[1] - times[0]), 1e-3)
    w = int(round((window_ms / 1000.0) / hop_s))
    if w < 1:
        return hz.copy(), voicing.copy()
    if w % 2 == 0:
        w += 1
    smoothed = hz.copy().astype(np.float64)
    voiced = voicing >= 0.05
    pad = w // 2
    for i in range(n):
        if not voiced[i]:
            continue
        lo = max(0, i - pad)
        hi = min(n, i + pad + 1)
        win = hz[lo:hi]
        win_voiced = voiced[lo:hi]
        if win_voiced.sum() == 0:
            continue
        vals = win[win_voiced]
        smoothed[i] = float(np.median(vals))
    return smoothed.astype(hz.dtype), voicing.copy()


def trim_silence(audio: np.ndarray, sr: int, *,
                 db_threshold: float = -40.0,
                 margin_ms: float = 10.0,
                 frame_ms: float = 20.0) -> tuple[np.ndarray, float, float]:
    """G-6: strip leading/trailing silence (< db_threshold dB FS).

    Returns (trimmed_audio, leading_pad_s, trailing_pad_s) where the
    pad fields say how much was stripped (caller can shift downstream
    beat/note times by `leading_pad_s` to keep absolute timing aligned
    if they want).
    """
    if audio.size == 0:
        return audio, 0.0, 0.0
    if audio.ndim > 1:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    frame_n = max(int(sr * (frame_ms / 1000.0)), 1)
    n_frames = len(mono) // frame_n
    if n_frames < 2:
        return audio, 0.0, 0.0
    eps = 1e-12
    frame_db = np.empty(n_frames, dtype=np.float64)
    for k in range(n_frames):
        seg = mono[k * frame_n:(k + 1) * frame_n].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg * seg) + eps))
        frame_db[k] = 20.0 * np.log10(rms + eps)
    above = frame_db > db_threshold
    if not above.any():
        return audio, 0.0, 0.0
    first = int(np.argmax(above))
    last = int(n_frames - 1 - np.argmax(above[::-1]))
    margin_n = int(sr * (margin_ms / 1000.0))
    start = max(0, first * frame_n - margin_n)
    end = min(len(mono), (last + 1) * frame_n + margin_n)
    leading_s = start / float(sr)
    trailing_s = (len(mono) - end) / float(sr)
    if audio.ndim > 1:
        return audio[:, start:end], leading_s, trailing_s
    return audio[start:end], leading_s, trailing_s
