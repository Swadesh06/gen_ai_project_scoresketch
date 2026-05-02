"""Pipeline configuration. Modes are exhaustive — soft | medium | hard."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

InputKind = Literal["humming", "instrument", "piano", "guitar"]
Mode = Literal["soft", "medium", "hard"]
PitchModel = Literal["pesto", "crepe"]
Transcriber = Literal["bytedance_piano", "basic_pitch"]


@dataclass
class ModeConfig:
    voicing_threshold: float
    min_note_seconds: float
    onset_merge_seconds: float
    dp_offgrid_penalty: float
    pitch_smooth_window: int

    @staticmethod
    def for_mode(mode: Mode) -> "ModeConfig":
        if mode == "soft":
            return ModeConfig(
                voicing_threshold=0.30,
                min_note_seconds=0.06,
                onset_merge_seconds=0.08,
                dp_offgrid_penalty=0.5,
                pitch_smooth_window=7,
            )
        if mode == "medium":
            return ModeConfig(
                voicing_threshold=0.50,
                min_note_seconds=0.10,
                onset_merge_seconds=0.05,
                dp_offgrid_penalty=1.0,
                pitch_smooth_window=5,
            )
        if mode == "hard":
            return ModeConfig(
                voicing_threshold=0.70,
                min_note_seconds=0.15,
                onset_merge_seconds=0.03,
                dp_offgrid_penalty=2.0,
                pitch_smooth_window=3,
            )
        raise ValueError(f"unknown mode: {mode!r}")


def default_transcriber(kind: InputKind) -> Transcriber:
    if kind == "humming":
        return "basic_pitch"
    if kind == "instrument":
        return "basic_pitch"
    if kind == "piano":
        return "bytedance_piano"
    if kind == "guitar":
        return "basic_pitch"
    raise ValueError(f"unknown input_kind: {kind!r}")


@dataclass
class PipelineConfig:
    input_kind: InputKind = "humming"
    mode: Mode = "soft"
    pitch_model: PitchModel = "pesto"
    transcriber: Transcriber | None = None
    tatums_per_beat: int = 12
    sample_rate: int = 22050
    render_svg: bool = True
    musicxml_path: str | None = None
    svg_path: str | None = None
    mode_config: ModeConfig = field(init=False)

    def __post_init__(self) -> None:
        self.mode_config = ModeConfig.for_mode(self.mode)
        if self.transcriber is None:
            self.transcriber = default_transcriber(self.input_kind)

    def is_humming(self) -> bool:
        return self.input_kind == "humming"
