"""Pipeline configuration. Modes are exhaustive — soft | medium | hard."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

InputKind = Literal["humming", "instrument", "piano", "guitar"]
Mode = Literal["soft", "medium", "hard"]
PitchModel = Literal["pesto", "crepe", "pesto_crepevoicing"]
Transcriber = Literal["bytedance_piano", "basic_pitch", "auto_piano", "yourmt3plus"]
NoteSegmenter = Literal["voicing", "hmm"]


@dataclass
class ModeConfig:
    voicing_threshold: float
    min_note_seconds: float
    onset_merge_seconds: float
    dp_offgrid_penalty: float
    pitch_smooth_window: int

    @staticmethod
    def for_mode(mode: Mode, pitch_model: "PitchModel" = "pesto") -> "ModeConfig":
        # voicing_threshold and pitch_smooth_window depend on the pitch_model
        # because PESTO confidence and CREPE periodicity have different scales.
        # Defaults tuned by exp_B2/B22 (PESTO) and exp_B36/B36b (hybrid).
        if mode == "soft":
            if pitch_model == "pesto_crepevoicing":
                return ModeConfig(
                    voicing_threshold=0.75, min_note_seconds=0.052,
                    onset_merge_seconds=0.026, dp_offgrid_penalty=0.5,
                    pitch_smooth_window=19,
                )
            return ModeConfig(
                voicing_threshold=0.315, min_note_seconds=0.052,
                onset_merge_seconds=0.026, dp_offgrid_penalty=0.5,
                pitch_smooth_window=15,
            )
        if mode == "medium":
            return ModeConfig(
                voicing_threshold=0.50, min_note_seconds=0.10,
                onset_merge_seconds=0.05, dp_offgrid_penalty=1.0,
                pitch_smooth_window=5,
            )
        if mode == "hard":
            return ModeConfig(
                voicing_threshold=0.70, min_note_seconds=0.15,
                onset_merge_seconds=0.03, dp_offgrid_penalty=2.0,
                pitch_smooth_window=3,
            )
        raise ValueError(f"unknown mode: {mode!r}")


def default_transcriber(kind: InputKind) -> Transcriber:
    if kind == "humming":
        return "basic_pitch"
    if kind == "instrument":
        return "basic_pitch"
    if kind == "piano":
        # B+2 item 2 (B63): YourMT3+ via auto_piano wins +3.9pp on Bach Fugues and
        # +12.6pp on 3-Romantic mean (ex-Liszt) vs ByteDance. Default to it.
        # Set transcriber="bytedance_piano" explicitly for the older fast path.
        return "auto_piano"
    if kind == "guitar":
        return "basic_pitch"
    raise ValueError(f"unknown input_kind: {kind!r}")


PerVoiceDP = Literal["auto", "on", "off"]
OctaveSanity = Literal["auto", "off"]


@dataclass
class PipelineConfig:
    input_kind: InputKind = "humming"
    mode: Mode = "soft"
    pitch_model: PitchModel = "pesto"
    transcriber: Transcriber | None = None
    note_segmenter: NoteSegmenter = "voicing"
    # Phase E item 7 ME-14 + ME-14-ext findings:
    # - tpb=24 (original):           mean MV2H 0.5377 (with octave_sanity)
    # - tpb=12 (production):         mean MV2H 0.5492 (+0.0115 over tpb=24)
    # - tpb=8 (ME-14-ext winner):    mean MV2H 0.5517 (+0.0025 over tpb=12)
    # - tpb=16 (ME-14-ext runner-up):mean MV2H 0.5515
    # tpb=12 is the production default — keeps integer ratio to render_tpb=12
    # (no resampling) at the cost of 0.0025 MV2H below the tpb=8 optimum.
    # Snap-F1 backwards-compat callers can override to 24 explicitly.
    tatums_per_beat: int = 12
    render_tpb: int = 12
    estimate_key: bool = True
    # Phase E item 7 ME-9: line-of-fifths enharmonic spelling on the rendered
    # score. Renderer-side polish only — does not change pitches or durations,
    # so no F1/snap risk. Default off for now; flip to True after ME-9
    # validation in `reports/me9_line_of_fifths.md` shows no regressions.
    enharmonic_spelling: bool = False
    # Phase D B79/B80: independent DP per voice + B76 transformer voice tracker.
    # auto: detect melody+accomp pieces (pitch IQR + density) and route there.
    # on: always use per-voice DP (still requires checkpoints/voice_transformer_b76/best.pt).
    # off: keep production shared-DP / greedy voice tracker behavior.
    per_voice_dp: PerVoiceDP = "auto"
    # Phase F-1: beat-tempo octave sanity check. "auto" runs the notes-per-
    # beat + fast-tempo/slow-note heuristic and halves/doubles beats when
    # the octave is clearly wrong. +0.088 MV2H on Bach BWV 856, +0.002 on
    # Chopin Berceuse, no change on the other 7 ASAP pieces. "off" disables.
    octave_sanity: OctaveSanity = "auto"
    sample_rate: int = 22050
    render_svg: bool = True
    musicxml_path: str | None = None
    svg_path: str | None = None
    mode_config: ModeConfig = field(init=False)

    def __post_init__(self) -> None:
        self.mode_config = ModeConfig.for_mode(self.mode, self.pitch_model)
        if self.transcriber is None:
            self.transcriber = default_transcriber(self.input_kind)

    def is_humming(self) -> bool:
        return self.input_kind == "humming"
