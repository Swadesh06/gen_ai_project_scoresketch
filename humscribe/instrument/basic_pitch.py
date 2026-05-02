"""Spotify Basic Pitch wrapper. Returns list[NoteEvent]."""
from __future__ import annotations
from pathlib import Path

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

from humscribe.notes import NoteEvent


def transcribe_basic_pitch(
    audio_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: float = 60.0,
) -> list[NoteEvent]:
    _, _, note_events = predict(
        str(audio_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
    )
    out: list[NoteEvent] = []
    for ev in note_events:
        on, off, midi_pitch, vel, _bends = _unpack(ev)
        out.append(NoteEvent(
            onset_s=float(on),
            offset_s=float(off),
            pitch_midi=int(midi_pitch),
            velocity=int(round(vel * 127)) if 0.0 <= vel <= 1.0 else int(vel),
        ))
    out.sort(key=lambda x: x.onset_s)
    return out


def _unpack(ev: tuple) -> tuple:
    if len(ev) >= 5:
        return ev[0], ev[1], ev[2], ev[3], ev[4]
    if len(ev) == 4:
        return ev[0], ev[1], ev[2], ev[3], None
    raise ValueError(f"unexpected basic_pitch note_event arity: {len(ev)}")
