"""NoteEvent dataclass — the common currency between stages."""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class NoteEvent:
    onset_s: float
    offset_s: float
    pitch_hz: float | None = None
    pitch_midi: int | None = None
    velocity: int = 80
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s

    def midi(self) -> int:
        if self.pitch_midi is not None:
            return int(self.pitch_midi)
        if self.pitch_hz is None or self.pitch_hz <= 0:
            return 0
        return int(round(69 + 12 * math.log2(self.pitch_hz / 440.0)))


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
