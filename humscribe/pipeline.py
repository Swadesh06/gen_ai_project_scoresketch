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
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import quantize_with_voice_tracking
from humscribe.rhythm.voice_transformer import get_b76_assigner, is_b76_available
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
    # Phase G G-6: trim leading/trailing silence so beat_this doesn't place beats
    # in silence. Humming only. The trim shifts downstream timing by `lead_s`;
    # we forward this through to beat positions later.
    lead_s, trail_s = 0.0, 0.0
    beat_audio_path = audio_path
    if cfg.is_humming() and cfg.silent_trim_g6 == "auto":
        from humscribe.post_process import trim_silence
        trimmed, lead_s, trail_s = trim_silence(audio, sr, db_threshold=cfg.silent_trim_db)
        if lead_s > 0.0 or trail_s > 0.0:
            import tempfile, soundfile  # type: ignore[import-untyped]
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            soundfile.write(tf.name, trimmed, sr)
            beat_audio_path = tf.name
    notes = _branch_notes(audio_path, audio, sr, cfg)
    notes = _filter_short_notes(notes, cfg.mode_config.min_note_seconds)
    # Phase G G-4: merge consecutive same-pitch NoteEvents (humming branch).
    if cfg.is_humming() and cfg.same_pitch_merge == "auto" and len(notes) > 1:
        from humscribe.post_process import merge_same_pitch
        notes = merge_same_pitch(notes, gap_s=cfg.same_pitch_merge_ms / 1000.0)
    # Phase F-2e: snap heuristic offsets to BiLSTM peaks when the model is
    # confident. Vocadito offset20 F1 0.343 → 0.370 at min_prob=0.30,
    # search_ms=50. Only applies to humming branch.
    if cfg.is_humming() and cfg.formant_offset_corrector == "auto":
        from humscribe.pitch.formant_corrector import correct_offsets
        notes = correct_offsets(notes, audio, sr)
    # B88 fix for B87 finding: beat_this without target_bpm hits half/double
    # tempo octaves on ~40% of pieces (BWV 846 → 60 instead of 120; BWV 856
    # → 230 instead of 115). Target 110 is a reasonable median for piano (B13
    # used per-piece score-derived targets in benchmarks). Pieces with truly
    # extreme tempos (Liszt fast passages) keep their detected octave because
    # the log2 distance to 110 is minimised in the correct octave.
    # Allow UI-side BPM hint to override the B88 default of 110.0.
    eff_target_bpm = float(cfg.target_bpm) if cfg.target_bpm else 110.0
    beats, downbeats, bpm = track_beats_beat_this(beat_audio_path, target_bpm=eff_target_bpm)
    if lead_s > 0.0 and len(beats) > 0:
        beats = beats + lead_s
        downbeats = downbeats + lead_s
    # Phase F-1: octave sanity check. Beat_this's target_bpm=110 covers most
    # cases but mis-octaves on slow pieces (Chopin Berceuse: detected 120
    # vs true 40) and dense fast pieces (Bach 856: detected 81 vs true
    # 240). The notes-per-beat heuristic catches both.
    if cfg.octave_sanity != "off" and not cfg.is_humming():
        from humscribe.beat.octave_sanity import (
            detect_octave_misalignment, apply_octave_correction,
        )
        diag = detect_octave_misalignment(beats, notes)
        if diag["recommend"] != "keep":
            beats, downbeats = apply_octave_correction(
                beats, downbeats, diag["recommend"]
            )
            if len(beats) >= 2:
                ibis = np.diff(beats)
                ibis = ibis[(ibis > 0.01) & (ibis < 5.0)]
                if len(ibis) > 0:
                    bpm = 60.0 / float(np.median(ibis))
    onsets = np.array([n.onset_s for n in notes], dtype=np.float64)
    offsets = np.array([n.offset_s for n in notes], dtype=np.float64)
    if len(onsets) > 0 and len(beats) >= 2:
        if not cfg.is_humming():
            # B79 + B80: per_voice_dp + learned voice tracker wins ~+2pp on
            # melody+accompaniment pieces (Chopin Berceuse-style). Detect via
            # `_should_use_per_voice_dp` and route accordingly.
            use_pvd = _should_use_per_voice_dp(notes, cfg)
            assigner = (get_b76_assigner() if use_pvd and is_b76_available()
                          else None)
            q_on, q_off = quantize_with_voice_tracking(
                notes, beats, tatums_per_beat=cfg.tatums_per_beat,
                per_voice_dp=use_pvd, voice_assigner=assigner,
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
    time_sig = cfg.time_sig_override or _infer_time_signature(beats, downbeats)
    # Phase G G-11: render_tpb auto-detect — slow pieces (median IOI > 0.3 s)
    # rendered at tpb=12 produce sextuplets/triplets that the human reader
    # can't parse cheaply; tpb=8 stays on 8th/16th/triplet ground.
    render_tpb_eff = cfg.render_tpb
    if cfg.render_tpb_auto == "auto" and len(onsets) >= 4:
        sorted_onsets = np.sort(onsets)
        median_ioi = float(np.median(np.diff(sorted_onsets)))
        if median_ioi > 0.30 and cfg.render_tpb == 12:
            render_tpb_eff = 8
    s = build_stream(
        notes, bpm=bpm, time_sig=time_sig,
        tatum_onsets=q_on if len(onsets) > 0 else None,
        tatum_offsets=q_off if len(onsets) > 0 else None,
        tatums_per_beat=cfg.tatums_per_beat,
        render_tpb=render_tpb_eff,
        estimate_key=cfg.estimate_key,
        enharmonic_spelling=cfg.enharmonic_spelling,
        key_override=cfg.key_override,
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
        elif cfg.pitch_model == "pesto_crepevoicing":
            t, hz, vc = track_pitch_hybrid_voicing(audio, sr)
        else:
            raise ValueError(f"unknown pitch_model: {cfg.pitch_model!r}")
        # Phase G G-5: 250 ms voiced-only median smoothing on the pitch trace
        # before segmentation. Mauch & Dixon 2014 pYIN published practice.
        if cfg.median_smooth_g5 == "auto":
            from humscribe.post_process import median_smooth_pitch
            hz, vc = median_smooth_pitch(t, hz, vc, window_ms=cfg.median_smooth_window_ms)
        return segment_pitch_to_notes(t, hz, vc, cfg.mode_config)
    if cfg.transcriber == "bytedance_piano":
        return transcribe_piano(audio_path)
    if cfg.transcriber == "basic_pitch":
        return transcribe_basic_pitch(audio_path)
    if cfg.transcriber == "yourmt3plus":
        # B+2 item 2: T5 seq2seq, broadest generalization.
        from humscribe.instrument.yourmt3plus import transcribe_yourmt3plus
        return transcribe_yourmt3plus(audio_path)
    if cfg.transcriber == "auto_piano":
        # B+2 item 2 (B63 result): YourMT3+ wins on every ASAP piece except Liszt
        # (which is structurally DP-bound per B54, both transcribers fail).
        # 9-piece mean snap: YMT3+ 0.774 vs ByteDance 0.713 (+6.1pp). 5-Bach mean:
        # YMT3+ 0.898 vs BD 0.859 (+3.9pp). 3 Romantic ex-Liszt mean: YMT3+ 0.806
        # vs BD 0.680 (+12.6pp). Beethoven 0.897 clears the ≥ 0.85 decision threshold.
        # Heuristic-based routing was unreliable (B61) — make YMT3+ unconditional.
        from humscribe.instrument.yourmt3plus import transcribe_yourmt3plus
        return transcribe_yourmt3plus(audio_path)
    raise ValueError(f"unknown transcriber: {cfg.transcriber!r}")


def _should_use_per_voice_dp(notes: list[NoteEvent], cfg: PipelineConfig) -> bool:
    """B79/B80: per_voice_dp + B76 voice tracker wins ~+2pp on
    melody+accompaniment textures (e.g. Chopin Berceuse) but ties or slightly
    regresses on dense polyphony (Schumann Toccata, Beethoven Sonata 21-1).

    Routing heuristic: opt in for piano-input pieces with low note-density
    (≤ 4 notes/sec) AND modest pitch-IQR (< 24 semitones), where
    melody+accompaniment hand separation is clean enough for the per-voice DP
    to add value. Dense Romantic chordal textures keep the original shared-DP
    path. User override is via the explicit `cfg.per_voice_dp` flag.
    """
    if cfg.per_voice_dp == "off":
        return False
    if cfg.per_voice_dp == "on":
        return True
    # auto: detect melody+accomp signature. Empirical thresholds from B79 data:
    #   Chopin Berceuse (winner +1.66pp): nps=7.97, iqr=17
    #   Liszt Sonata (≈neutral -0.07pp):  nps=10.3, iqr=22
    #   Beethoven 21-1 (≈neutral -0.36pp): nps=14.0, iqr=26
    #   Schumann Toccata (lost -3.9pp):    nps=21.4, iqr=17
    #   Bach BWV 854 (fugue, untested):    nps=13.3, iqr=14
    # Picking nps < 10 isolates Chopin (the proven winner) and excludes the
    # losers. Phase E: tune with more pieces.
    if cfg.input_kind not in {"piano", "instrument"}:
        return False
    if len(notes) < 32:
        return False
    # Use min/max because some transcribers (ByteDance) return chord notes in
    # descending-pitch order rather than time order.
    onsets = [n.onset_s for n in notes]
    span = max(max(onsets) - min(onsets), 1e-3)
    notes_per_sec = len(notes) / span
    midis = [n.midi() for n in notes if n.midi() > 0]
    if not midis:
        return False
    pitch_iqr = float(np.percentile(midis, 75) - np.percentile(midis, 25))
    return notes_per_sec < 10.0 and pitch_iqr < 24


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
