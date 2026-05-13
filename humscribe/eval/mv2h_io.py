"""MV2H text-format conversion (Phase E item 1).

MV2H expects a line-based text file with:
- `Note pitch on onVal offVal voice` (all integers, ms)
- `Tatum time` (one per metric tatum)
- `Hierarchy bpb,sbpb tpsb a=al`
- `Key tonic maj/min`

We always emit `on == onVal` and `off == offVal` (the times are already
quantised by Cemgil DP upstream, or we use the `-a` non-aligned flag).

This module never invokes Java; it produces strings only.
"""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

import numpy as np

from humscribe.notes import NoteEvent


_KEY_LETTERS = "C C# D D# E F F# G G# A A# B".split()
_KEY_TO_INT = {ltr: i for i, ltr in enumerate(_KEY_LETTERS)}
_FLAT_TO_SHARP = {"D-": "C#", "E-": "D#", "G-": "F#", "A-": "G#", "B-": "A#"}


def _parse_time_signature(time_sig: str) -> tuple[int, int]:
    """'3/4' -> (3, 4). Returns (numerator, denominator)."""
    if "/" not in time_sig:
        return (4, 4)
    a, b = time_sig.split("/", 1)
    try:
        return (int(a), int(b))
    except ValueError:
        return (4, 4)


def _hierarchy_line(time_sig: str, tatums_per_subbeat: int = 1) -> str:
    """Construct the MV2H Hierarchy line from a time signature string.

    bpb = beats per bar = numerator.
    sbpb = subbeats per beat = 3 if compound (denominator==8 and num%3==0) else 2.
    tpsb = tatums per subbeat.
    al = anacrusis length (0 if unknown).
    """
    num, den = _parse_time_signature(time_sig)
    if den == 8 and num % 3 == 0 and num >= 6:
        sbpb = 3
        bpb = num // 3
    else:
        sbpb = 2
        bpb = num
    return f"Hierarchy {bpb},{sbpb} {int(tatums_per_subbeat)} a=0"


def _tatum_grid_ms(bpm: float, total_ms: int, tatums_per_beat: int) -> list[int]:
    """Tatum pulses spaced at 60_000 / (bpm * tatums_per_beat) ms, starting at 0,
    covering [0, total_ms]."""
    if bpm <= 0 or tatums_per_beat <= 0 or total_ms <= 0:
        return [0]
    step_ms = 60_000.0 / (float(bpm) * float(tatums_per_beat))
    if step_ms <= 0:
        return [0]
    n = int(total_ms // step_ms) + 1
    return [int(round(i * step_ms)) for i in range(n + 1)]


def _tatum_grid_from_beats(beats, total_ms: int, tatums_per_beat: int) -> list[int]:
    """Phase G G-2: tatum pulses interpolated linearly between real beat
    positions. Falls back to the last inter-beat interval to extend past the
    last beat up to `total_ms` so pred and GT cover the same window.

    `beats` is a sequence of beat times in seconds.
    """
    bs = [float(b) for b in beats]
    if len(bs) < 2 or tatums_per_beat <= 0 or total_ms <= 0:
        return [0]
    pulses: list[int] = []
    for k in range(len(bs) - 1):
        seg_dur_ms = (bs[k + 1] - bs[k]) * 1000.0
        if seg_dur_ms <= 0:
            continue
        for j in range(tatums_per_beat):
            t_ms = int(round(bs[k] * 1000.0 + j * seg_dur_ms / tatums_per_beat))
            if 0 <= t_ms <= total_ms:
                pulses.append(t_ms)
    # Extend past the last beat using the last ibi.
    last_ibi_ms = (bs[-1] - bs[-2]) * 1000.0 if len(bs) >= 2 else 500.0
    if last_ibi_ms > 0:
        t = bs[-1] * 1000.0
        # include the last beat itself as a tatum on the downbeat (j=0).
        if 0 <= int(round(t)) <= total_ms:
            pulses.append(int(round(t)))
        step_ms = last_ibi_ms / tatums_per_beat
        while t + step_ms <= total_ms:
            t += step_ms
            pulses.append(int(round(t)))
    pulses.sort()
    # Deduplicate consecutive identical positions caused by rounding.
    dedup = [pulses[0]] if pulses else [0]
    for p in pulses[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    return dedup


def notes_to_mv2h_format(
    notes: Sequence[NoteEvent],
    bpm: float,
    time_sig: str = "4/4",
    voices: Sequence[int] | None = None,
    *,
    tatums_per_beat: int = 4,
    key_tonic: int | None = None,
    key_mode: str = "Maj",
    quantise_to_tatum: bool = False,
    beats: Sequence[float] | None = None,
) -> str:
    """Convert (NoteEvent[], bpm, time_sig) -> MV2H text.

    `voices` is a parallel sequence of int voice IDs; defaults to all-zero.
    `tatums_per_beat` controls the tatum density we emit (not the pipeline DP
    resolution — MV2H uses tatums as the metric resolution proxy, denser is
    safer for the `-a` non-aligned evaluator).
    `quantise_to_tatum`: when True, emitted `onVal`/`offVal` snap to the
    nearest tatum pulse. Default False: produces higher MV2H scores because
    the DTW alignment in `-a` mode struggles when many predicted notes
    collapse onto the same onVal bucket. Item 6's sweep can flip this on
    if it finds a regime where it helps.
    """
    if voices is None:
        voices = [0] * len(notes)
    if len(voices) != len(notes):
        raise ValueError("voices length must match notes length")

    if bpm <= 0 or tatums_per_beat <= 0:
        tatum_ms = 250.0  # fallback: quarter-note tatum at 60 bpm
    else:
        tatum_ms = 60_000.0 / (float(bpm) * float(tatums_per_beat))

    def _q(ms: int) -> int:
        if not quantise_to_tatum or tatum_ms <= 0:
            return ms
        return int(round(ms / tatum_ms) * tatum_ms)

    lines: list[str] = []
    max_off_ms = 0
    for ev, v in zip(notes, voices):
        midi = ev.midi()
        if midi <= 0:
            continue
        on_ms = int(round(ev.onset_s * 1000.0))
        off_ms = int(round(ev.offset_s * 1000.0))
        if off_ms <= on_ms:
            off_ms = on_ms + 1
        on_val = _q(on_ms)
        off_val = _q(off_ms)
        if off_val <= on_val:
            off_val = on_val + max(int(tatum_ms), 1)
        max_off_ms = max(max_off_ms, off_ms)
        lines.append(f"Note {int(midi)} {on_ms} {on_val} {off_val} {int(v)}")
    if beats is not None and len(list(beats)) >= 2:
        # Phase G G-2: place tatums at real beat positions (interpolated) so
        # MV2H's meter sub-score compares position-accurate grids on both
        # sides. Uniform-from-bpm fallback collapses meter on tempo-rubato
        # pieces (Liszt) and any clip where beat_this's median IBI differs
        # from the GT's first tempo marking.
        tat_positions = _tatum_grid_from_beats(beats, max_off_ms, tatums_per_beat)
    else:
        tat_positions = _tatum_grid_ms(float(bpm), max_off_ms, tatums_per_beat)
    for t in tat_positions:
        lines.append(f"Tatum {t}")
    lines.append(_hierarchy_line(time_sig, tatums_per_subbeat=tatums_per_beat // 2 if tatums_per_beat >= 2 else 1))
    if key_tonic is not None:
        mode_str = "Maj" if str(key_mode).lower().startswith("maj") else "min"
        tonic = int(key_tonic) % 12
        lines.append(f"Key {tonic} {mode_str}")
    return "\n".join(lines) + "\n"


def stream_to_mv2h_format(stream_or_score, *, tatums_per_beat: int = 4) -> str:
    """music21 Stream/Score -> MV2H text.

    Voices are discovered by walking music21 Part/Voice containers.
    Tempo and time signature are read from the stream; defaults if missing.
    """
    from music21 import stream as m21stream
    from music21 import tempo as m21tempo
    from music21 import meter as m21meter
    from music21 import note as m21note
    from music21 import chord as m21chord
    from music21 import key as m21key

    s = stream_or_score
    bpm_marks = list(s.recurse().getElementsByClass(m21tempo.MetronomeMark))
    bpm = float(bpm_marks[0].number) if bpm_marks and bpm_marks[0].number else 120.0
    ts_list = list(s.recurse().getElementsByClass(m21meter.TimeSignature))
    time_sig = f"{ts_list[0].numerator}/{ts_list[0].denominator}" if ts_list else "4/4"
    key_tonic, key_mode = _key_from_stream(s, m21key)

    notes: list[NoteEvent] = []
    voices: list[int] = []
    voice_id = 0
    parts = list(s.parts) if isinstance(s, m21stream.Score) else [s]
    for part in parts:
        sub_voices = list(part.recurse().getElementsByClass(m21stream.Voice))
        if sub_voices:
            for sv in sub_voices:
                _collect_notes_from_container(sv, voice_id, bpm, notes, voices, m21note, m21chord)
                voice_id += 1
        else:
            _collect_notes_from_container(part, voice_id, bpm, notes, voices, m21note, m21chord)
            voice_id += 1
    return notes_to_mv2h_format(
        notes, bpm=bpm, time_sig=time_sig, voices=voices,
        tatums_per_beat=tatums_per_beat,
        key_tonic=key_tonic, key_mode=key_mode,
    )


def _collect_notes_from_container(container, voice_id: int, bpm: float,
                                  notes: list[NoteEvent], voices: list[int],
                                  m21note, m21chord) -> None:
    """Append (NoteEvent, voice) pairs from a music21 Stream/Voice container."""
    sec_per_ql = 60.0 / max(bpm, 1e-3)
    for el in container.recurse().notes:
        on_s = float(el.offset) * sec_per_ql
        dur_s = float(el.quarterLength) * sec_per_ql
        off_s = on_s + max(dur_s, 1e-3)
        if isinstance(el, m21chord.Chord):
            for p in el.pitches:
                notes.append(NoteEvent(onset_s=on_s, offset_s=off_s,
                                       pitch_midi=int(p.midi), velocity=80))
                voices.append(voice_id)
        elif isinstance(el, m21note.Note):
            notes.append(NoteEvent(onset_s=on_s, offset_s=off_s,
                                   pitch_midi=int(el.pitch.midi), velocity=80))
            voices.append(voice_id)


def _key_from_stream(s, m21key) -> tuple[int | None, str]:
    ks_list = list(s.recurse().getElementsByClass(m21key.KeySignature))
    if not ks_list:
        return None, "Maj"
    k = ks_list[0]
    if isinstance(k, m21key.Key):
        tonic_name = k.tonic.name
        mode = "maj" if k.mode == "major" else "min"
    else:
        tonic_name = _major_tonic_from_sharps(int(k.sharps))
        mode = "maj"
    tonic_name = _FLAT_TO_SHARP.get(tonic_name, tonic_name)
    return _KEY_TO_INT.get(tonic_name, 0), mode


def _major_tonic_from_sharps(sharps: int) -> str:
    # Major keys ordered by circle of fifths from -7..+7 sharps.
    order = ["C-", "G-", "D-", "A-", "E-", "B-", "F",
             "C", "G", "D", "A", "E", "B", "F#", "C#"]
    idx = max(0, min(len(order) - 1, sharps + 7))
    return order[idx]


def score_to_mv2h_format(score, *, tatums_per_beat: int = 4) -> str:
    """Alias for stream_to_mv2h_format kept for v3 spec naming compatibility."""
    return stream_to_mv2h_format(score, tatums_per_beat=tatums_per_beat)


def midi_to_mv2h_format(midi_path: str | Path, *, tatums_per_beat: int = 4) -> str:
    """Read a MIDI file via pretty_midi and emit MV2H text.

    Voice id = MIDI channel + 16 * track index (so cross-track polyphony is
    preserved while keeping ids small).
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    bpm = float(pm.estimate_tempo()) if pm.instruments else 120.0
    ts_changes = pm.time_signature_changes
    if ts_changes:
        ts0 = ts_changes[0]
        time_sig = f"{ts0.numerator}/{ts0.denominator}"
    else:
        time_sig = "4/4"
    ks_changes = pm.key_signature_changes
    key_tonic = None
    key_mode = "Maj"
    if ks_changes:
        ks0 = ks_changes[0]
        # pretty_midi key_number: 0..11 major, 12..23 minor
        kn = int(ks0.key_number)
        if kn >= 12:
            key_tonic = kn - 12
            key_mode = "min"
        else:
            key_tonic = kn
            key_mode = "Maj"

    notes: list[NoteEvent] = []
    voices: list[int] = []
    for t_idx, inst in enumerate(pm.instruments):
        v = t_idx
        for n in inst.notes:
            notes.append(NoteEvent(onset_s=float(n.start), offset_s=float(n.end),
                                   pitch_midi=int(n.pitch), velocity=int(n.velocity)))
            voices.append(v)
    return notes_to_mv2h_format(
        notes, bpm=bpm, time_sig=time_sig, voices=voices,
        tatums_per_beat=tatums_per_beat,
        key_tonic=key_tonic, key_mode=key_mode,
    )


def musicxml_to_mv2h_format(xml_path: str | Path, *, tatums_per_beat: int = 4) -> str:
    """Read a MusicXML file via music21 and emit MV2H text."""
    from music21 import converter
    s = converter.parse(str(xml_path))
    return stream_to_mv2h_format(s, tatums_per_beat=tatums_per_beat)


def notes_with_voices_from_pipeline(result) -> tuple[list[NoteEvent], list[int]]:
    """Helper: pull (notes, voices) from a TranscribeResult.

    If the result didn't run voice tracking, every note is voice 0.
    """
    notes = list(result.notes)
    voices = [0] * len(notes)
    return notes, voices
