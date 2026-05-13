"""Phase G G-14: multi-take consensus transcription.

User records N takes of the same melody. We transcribe each, then keep
notes that appear in >= ceil(N/2) takes within +-50 ms of one another.
The first take's tatum grid + key signature is reused for the merged
output so the score remains coherent.

This is a thin wrapper around pipeline.transcribe; the consensus logic
matches notes across takes with greedy onset alignment.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from humscribe.config import PipelineConfig
from humscribe.notes import NoteEvent
from humscribe.pipeline import TranscribeResult, transcribe


def _match_notes(ref: Sequence[NoteEvent], cand: Sequence[NoteEvent],
                 onset_tol_s: float = 0.05, pitch_tol: int = 1) -> list[bool]:
    """Greedy onset-time match. Returns a parallel bool list to ref."""
    used = [False] * len(cand)
    out = [False] * len(ref)
    for i, r in enumerate(ref):
        best = -1; best_dt = onset_tol_s + 1.0
        for j, c in enumerate(cand):
            if used[j]:
                continue
            if r.pitch_midi is None or c.pitch_midi is None:
                continue
            if abs(int(c.pitch_midi) - int(r.pitch_midi)) > pitch_tol:
                continue
            dt = abs(float(c.onset_s) - float(r.onset_s))
            if dt <= onset_tol_s and dt < best_dt:
                best_dt = dt; best = j
        if best >= 0:
            used[best] = True
            out[i] = True
    return out


def consensus_transcribe(audio_paths: Sequence[str | Path],
                          cfg: PipelineConfig | None = None,
                          *, onset_tol_s: float = 0.05,
                          pitch_tol: int = 1) -> TranscribeResult:
    """Transcribe each take, return a result whose `notes` are the consensus."""
    if not audio_paths:
        raise ValueError("consensus_transcribe needs at least one audio path")
    results = [transcribe(str(p), cfg) for p in audio_paths]
    if len(results) == 1:
        return results[0]
    primary = results[0]
    keep_flags = [True] * len(primary.notes)
    threshold = math.ceil(len(results) / 2)
    counts = [1] * len(primary.notes)
    for r in results[1:]:
        matches = _match_notes(primary.notes, r.notes,
                                onset_tol_s=onset_tol_s, pitch_tol=pitch_tol)
        for i, m in enumerate(matches):
            if m:
                counts[i] += 1
    consensus = [primary.notes[i] for i in range(len(primary.notes))
                 if counts[i] >= threshold]
    return TranscribeResult(
        notes=consensus, beats=primary.beats, downbeats=primary.downbeats,
        bpm=primary.bpm, musicxml=primary.musicxml, svg=primary.svg,
        tatum_onsets=primary.tatum_onsets, tatum_offsets=primary.tatum_offsets,
    )
