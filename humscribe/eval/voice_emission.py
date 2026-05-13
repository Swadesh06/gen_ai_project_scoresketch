"""Phase G G-1: voice IDs for MV2H emission.

The pipeline already produces voice assignments via the B76 Transformer
voice tracker (piano/instrument input) or greedy voice tracking
(fallback). Until Phase G the MV2H emitter discarded these and passed
voice=0 for every note, capping the MV2H voice sub-score at the
"single voice" default for any multi-voice GT.

This module computes per-note voice IDs in the canonical order
expected by `notes_to_mv2h_format` (parallel to the `notes` list).
"""
from __future__ import annotations
from typing import Sequence

from humscribe.notes import NoteEvent


def _voices_from_partition(partition: list[list[int]], n: int) -> list[int]:
    """Convert a partition (list of voice -> note indices) to per-note voice IDs.

    Notes not present in any partition get voice 0; partition voice index is
    used as the MV2H integer voice id.
    """
    out = [0] * n
    for v_id, idxs in enumerate(partition):
        for i in idxs:
            if 0 <= i < n:
                out[i] = v_id
    return out


def voice_ids_greedy(notes: Sequence[NoteEvent]) -> list[int]:
    """Fallback: greedy voice assignment (humscribe.rhythm.voice_tracking)."""
    from humscribe.rhythm.voice_tracking import (
        VoiceTrackConfig, adaptive_pitch_jump, assign_voices,
    )
    if not notes:
        return []
    cfg = VoiceTrackConfig(pitch_jump=adaptive_pitch_jump(notes))
    partition = assign_voices(notes, cfg)
    return _voices_from_partition(partition, len(notes))


def voice_ids_b76(notes: Sequence[NoteEvent]) -> list[int] | None:
    """Run the B76 Transformer voice tracker. Returns None if unavailable."""
    from humscribe.rhythm.voice_transformer import (
        get_b76_assigner, is_b76_available,
    )
    if not is_b76_available():
        return None
    if not notes:
        return []
    try:
        assigner = get_b76_assigner()
    except Exception:
        return None
    try:
        partition = assigner(notes)
    except Exception:
        return None
    return _voices_from_partition(partition, len(notes))


def voice_ids_for_emission(notes: Sequence[NoteEvent], input_kind: str = "piano") -> list[int]:
    """Return per-note voice IDs for MV2H emission.

    Routing:
    - humming: monophonic -> all zeros (no voice info to surface).
    - piano/instrument: B76 if available, else greedy fallback.
    """
    n = len(notes)
    if n == 0:
        return []
    if input_kind == "humming":
        return [0] * n
    v = voice_ids_b76(notes)
    if v is not None:
        return v
    return voice_ids_greedy(notes)
