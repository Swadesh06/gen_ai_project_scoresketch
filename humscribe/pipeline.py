"""Top-level transcribe entrypoint. Wires Stages 1-6 per DESIGN_NOTES.md."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.config import PipelineConfig
from humscribe.instrument.basic_pitch import transcribe_basic_pitch
from humscribe.instrument.piano import transcribe_piano
from humscribe.notes import NoteEvent
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import quantize_with_voice_tracking
from humscribe.score import build_stream, render_svg, write_musicxml


@dataclass
class TranscribeResult:
    notes: list[NoteEvent]
    beats: np.ndarray
    downbeats: np.ndarray
    bpm: float
    musicxml: str
    svg: str
    tatum_onsets: np.ndarray
    tatum_offsets: np.ndarray

    @property
    def n_notes(self) -> int:
        return len(self.notes)


def transcribe(audio_path: str, cfg: PipelineConfig | None = None) -> TranscribeResult:
    cfg = cfg or PipelineConfig()
    audio, sr = load_audio(audio_path, target_sr=cfg.sample_rate)
    notes = _branch_notes(audio_path, audio, sr, cfg)
    notes = _filter_short_notes(notes, cfg.mode_config.min_note_seconds)
    beats, downbeats, bpm = track_beats_beat_this(audio_path)
    onsets = np.array([n.onset_s for n in notes], dtype=np.float64)
    offsets = np.array([n.offset_s for n in notes], dtype=np.float64)
    if len(onsets) > 0 and len(beats) >= 2:
        if not cfg.is_humming():
            q_on, q_off = quantize_with_voice_tracking(
                notes, beats, tatums_per_beat=cfg.tatums_per_beat,
            )
        else:
            q_on, q_off = viterbi_quantize_rhythm(
                onsets, offsets, beats,
                tatums_per_beat=cfg.tatums_per_beat,
                offgrid_penalty=cfg.mode_config.dp_offgrid_penalty,
            )
    else:
        q_on = np.zeros(len(onsets), dtype=np.int64)
        q_off = np.zeros(len(onsets), dtype=np.int64)
    time_sig = _infer_time_signature(beats, downbeats)
    s = build_stream(
        notes, bpm=bpm, time_sig=time_sig,
        tatum_onsets=q_on if len(onsets) > 0 else None,
        tatum_offsets=q_off if len(onsets) > 0 else None,
        tatums_per_beat=cfg.tatums_per_beat,
    )
    musicxml = write_musicxml(s, cfg.musicxml_path)
    svg = render_svg(s, notes, bpm) if cfg.render_svg else ""
    if cfg.svg_path:
        Path(cfg.svg_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.svg_path).write_text(svg)
    return TranscribeResult(
        notes=notes, beats=beats, downbeats=downbeats, bpm=bpm,
        musicxml=musicxml, svg=svg,
        tatum_onsets=q_on, tatum_offsets=q_off,
    )


def _branch_notes(audio_path: str, audio: np.ndarray, sr: int, cfg: PipelineConfig) -> list[NoteEvent]:
    if cfg.is_humming():
        if cfg.pitch_model == "pesto":
            t, hz, vc = track_pitch_pesto(audio, sr)
        elif cfg.pitch_model == "crepe":
            t, hz, vc = track_pitch_crepe(audio, sr)
        else:
            raise ValueError(f"unknown pitch_model: {cfg.pitch_model!r}")
        return segment_pitch_to_notes(t, hz, vc, cfg.mode_config)
    if cfg.transcriber == "bytedance_piano":
        return transcribe_piano(audio_path)
    if cfg.transcriber == "basic_pitch":
        return transcribe_basic_pitch(audio_path)
    raise ValueError(f"unknown transcriber: {cfg.transcriber!r}")


def _filter_short_notes(notes: list[NoteEvent], min_s: float) -> list[NoteEvent]:
    return [n for n in notes if (n.offset_s - n.onset_s) >= min_s]


def _infer_time_signature(beats: np.ndarray, downbeats: np.ndarray) -> str:
    if len(downbeats) < 2 or len(beats) < 2:
        return "4/4"
    avg_db = float(np.mean(np.diff(downbeats)))
    avg_b = float(np.mean(np.diff(beats)))
    if avg_b <= 0:
        return "4/4"
    bpd = max(int(round(avg_db / avg_b)), 2)
    if bpd == 3:
        return "3/4"
    if bpd == 6:
        return "6/8"
    if bpd == 2:
        return "2/4"
    return "4/4"
