"""ByteDance piano transcription wrapper. Returns list[NoteEvent].

First call downloads ~170 MB Zenodo checkpoint to ~/piano_transcription_inference_data/.
Defaults to CUDA when available; CPU is the fallback.
"""
from __future__ import annotations
from pathlib import Path
import tempfile

import librosa
import numpy as np
import torch

from piano_transcription_inference import PianoTranscription, sample_rate as PT_SR

from humscribe.notes import NoteEvent


_CACHED: dict[str, "PianoTranscription"] = {}


def _autodevice() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(device: str | None = None) -> "PianoTranscription":
    dev = device or _autodevice()
    if dev not in _CACHED:
        _CACHED[dev] = PianoTranscription(device=dev)
    return _CACHED[dev]


def transcribe_piano(audio_path: str, device: str | None = None) -> list[NoteEvent]:
    audio, _ = librosa.load(str(audio_path), sr=PT_SR, mono=True)
    audio = audio.astype(np.float32)
    model = _get_model(device=device)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        midi_out = tf.name
    res = model.transcribe(audio, midi_out)
    return _events_from_result(res, midi_out)


def _events_from_result(res: dict | None, midi_path: str) -> list[NoteEvent]:
    notes_field = (res or {}).get("est_note_events")
    if notes_field:
        out: list[NoteEvent] = []
        for n in notes_field:
            out.append(NoteEvent(
                onset_s=float(n["onset_time"]),
                offset_s=float(n["offset_time"]),
                pitch_midi=int(n["midi_note"]),
                velocity=int(n.get("velocity", 80)),
            ))
        return out
    return _events_from_midi(midi_path)


def _events_from_midi(midi_path: str) -> list[NoteEvent]:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    out: list[NoteEvent] = []
    for inst in pm.instruments:
        for n in inst.notes:
            out.append(NoteEvent(
                onset_s=float(n.start),
                offset_s=float(n.end),
                pitch_midi=int(n.pitch),
                velocity=int(n.velocity),
            ))
    out.sort(key=lambda x: x.onset_s)
    return out
