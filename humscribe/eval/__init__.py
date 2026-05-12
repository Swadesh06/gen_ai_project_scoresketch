"""Phase E evaluation utilities (item 1: MV2H end-to-end score-similarity)."""
from humscribe.eval.mv2h import compute_mv2h, compute_mv2h_from_streams, MV2HResult
from humscribe.eval.mv2h_io import (
    notes_to_mv2h_format,
    stream_to_mv2h_format,
    score_to_mv2h_format,
    midi_to_mv2h_format,
    musicxml_to_mv2h_format,
)

__all__ = [
    "compute_mv2h",
    "compute_mv2h_from_streams",
    "MV2HResult",
    "notes_to_mv2h_format",
    "stream_to_mv2h_format",
    "score_to_mv2h_format",
    "midi_to_mv2h_format",
    "musicxml_to_mv2h_format",
]
