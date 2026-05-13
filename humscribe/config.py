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
FormantOffsetCorrector = Literal["auto", "off"]

# Phase G: emission and post-processing flags.
SamePitchMerge = Literal["auto", "off"]
MedianSmoothG5 = Literal["auto", "off"]
SilentTrimG6 = Literal["auto", "off"]


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
    # Phase G G-11: auto-downgrade render_tpb from 12 to 8 when the median
    # note IOI > 0.3 s (slow piece — fewer subdivisions in the score). Two-
    # pass tuplet counting is expensive; we approximate by IOI threshold.
    render_tpb_auto: Literal["auto", "off"] = "auto"
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
    # Phase F-2e: BiLSTM-based offset corrector for humming. When "auto",
    # the formant_offset_vocadito fold-0 checkpoint is loaded once and snaps
    # heuristic offsets to BiLSTM peaks within ±50 ms when the BiLSTM prob
    # ≥ 0.30. Lifts Vocadito offset20 F1 by +0.027 (0.343 → 0.370). Only
    # applied on humming input (not piano/instrument). Default "off" — the
    # checkpoint is Vocadito-trained and the per-piece worst-case
    # regression (-0.135 on voc_38) is too risky for default-on.
    formant_offset_corrector: FormantOffsetCorrector = "off"
    # Phase G G-4: merge consecutive same-pitch NoteEvents within
    # `same_pitch_merge_ms` gap. CREPE Notes 2023 practice for vibrato-
    # fragmentation. Humming branch only.
    # **Default "auto"** — the G-4-isolated strict measurement on the
    # canonical `scripts/gate_vocadito_conp_phase_g.py --apply g4` over the
    # full 40-clip A1 corpus gave mean noff F1 = 0.6776 vs baseline 0.6652
    # (Δ +0.0124, clears the strict ≥ 0.67 threshold). The earlier
    # "G-4+G-5+G-6 combined" regression was driven by G-5, not G-4.
    same_pitch_merge: SamePitchMerge = "auto"
    same_pitch_merge_ms: float = 80.0
    # Phase G G-5: 250 ms voiced-only median smoothing on the pitch trace
    # before segmentation (Mauch & Dixon 2014 pYIN). **Default "off"** —
    # same canonical-gate regression as G-4. Kept behind opt-in flag.
    median_smooth_g5: MedianSmoothG5 = "off"
    median_smooth_window_ms: float = 250.0
    # Phase G G-6: strip leading/trailing silence below `silent_trim_db`
    # before beat_this so beats don't land in silence. Humming branch only.
    # **Default "off"** — Vocadito clips don't have > 100 ms leading silence
    # so G-6 is a no-op on the strict-test corpus and the time-shift bug in
    # the inline gate path (audio truncated for pitch tracking shifted
    # absolute onsets) caused catastrophic noff F1 regression when wired
    # blindly. The production pipeline path was unaffected (it only routes
    # trimmed audio into beat_this, not segmentation), but the default-off
    # state keeps the production and gate paths in lockstep.
    silent_trim_g6: SilentTrimG6 = "off"
    silent_trim_db: float = -40.0
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
