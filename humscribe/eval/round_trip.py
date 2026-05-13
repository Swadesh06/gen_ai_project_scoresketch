"""Phase G G-8: round-trip self-consistency metric (Cohen et al. 2020).

audio → pipeline.transcribe → MIDI → pretty_midi.synthesize → MFCC-DTW
distance to the original audio. Lower distance = better round-trip
match. Reference-free — needs no GT annotation, so it works on the
hundreds of unannotated clips in the dataset hierarchy and unlocks
unlabelled sweeps.

Implementation note: pretty_midi.PrettyMIDI.synthesize() emits a
sinusoidal mock instead of a soundfont render, so we capture
melody/rhythm correspondence and gloss over timbre. That is fine for
the self-consistency signal we want — exact timbre match isn't on the
table when one side is the original recording and the other is a
synthesised MIDI.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from humscribe.notes import NoteEvent


@dataclass
class RoundTripResult:
    distance: float
    mfcc_pred_frames: int
    mfcc_ref_frames: int


def _mfcc(audio: np.ndarray, sr: int, *, n_mfcc: int = 13, hop: int = 512) -> np.ndarray:
    import librosa
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return librosa.feature.mfcc(y=audio.astype(np.float32), sr=sr,
                                  n_mfcc=n_mfcc, hop_length=hop, n_fft=2048)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised DTW cost between two feature matrices.

    librosa expects feature-major layout X.shape=(K, N) where K is the
    feature dim and N is the time dim. Cosine cost is scale-invariant.
    Returns the bottom-right of the cumulative cost matrix divided by
    path length so the value is comparable across clip durations.
    """
    import librosa
    d, _wp = librosa.sequence.dtw(X=a, Y=b, metric="cosine", subseq=False)
    cost = float(d[-1, -1])
    return cost / max(d.shape[0] + d.shape[1], 1)


def notes_to_pretty_midi(notes: Sequence[NoteEvent], bpm: float = 120.0):
    """Build a single-instrument PrettyMIDI from NoteEvent list."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=0)
    for n in notes:
        m = n.midi()
        if m <= 0:
            continue
        inst.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, n.velocity))),
            pitch=int(max(0, min(127, m))),
            start=float(n.onset_s), end=float(max(n.offset_s, n.onset_s + 0.01)),
        ))
    pm.instruments.append(inst)
    return pm


def round_trip_distance(audio_ref: np.ndarray, sr_ref: int,
                         notes: Sequence[NoteEvent], bpm: float = 120.0,
                         *, synth_fs: int = 16000) -> RoundTripResult:
    """Synthesise `notes` at `synth_fs`, MFCC both sides, return DTW cost."""
    pm = notes_to_pretty_midi(notes, bpm=bpm)
    if not pm.instruments or not pm.instruments[0].notes:
        return RoundTripResult(distance=float("inf"), mfcc_pred_frames=0, mfcc_ref_frames=0)
    synth = pm.synthesize(fs=synth_fs).astype(np.float32)
    if synth.size == 0:
        return RoundTripResult(distance=float("inf"), mfcc_pred_frames=0, mfcc_ref_frames=0)
    import librosa
    if sr_ref != synth_fs:
        audio_ref_rs = librosa.resample(audio_ref.astype(np.float32),
                                          orig_sr=sr_ref, target_sr=synth_fs)
    else:
        audio_ref_rs = audio_ref.astype(np.float32)
    mref = _mfcc(audio_ref_rs, synth_fs)
    mpred = _mfcc(synth, synth_fs)
    # mref / mpred shape: (n_mfcc, T). DTW wants (K, N) so pass as-is.
    dist = _dtw_distance(mref, mpred)
    return RoundTripResult(distance=dist, mfcc_pred_frames=mpred.shape[1],
                            mfcc_ref_frames=mref.shape[1])
