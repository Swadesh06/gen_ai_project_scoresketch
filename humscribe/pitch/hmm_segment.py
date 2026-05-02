"""HMM/Viterbi note segmenter (Phase B Exp B4).

State space: 1 silent state + one state per integer semitone in `[midi_lo, midi_hi]`.
Observations per frame: `(midi_obs, voicing)` where `midi_obs = hz_to_midi(hz)`
when voicing > 0 else 0, and `voicing in [0, 1]`.

Emission log-likelihoods (Gaussian on midi error + Bernoulli-style on voicing):
- silent  : log P ∝ -((voicing - 0)^2 / (2*sigma_v^2))
- active_p: log P ∝ -((voicing - 1)^2 / (2*sigma_v^2))  -((midi_obs - p)^2 / (2*sigma_m^2))

Transition log-probabilities (rows sum to 1 in linear space):
- silent → silent: 1 - p_start
- silent → active_p: p_start / N_pitches  (uniform note start)
- active_p → silent: p_end
- active_p → active_p: p_sustain
- active_p → active_q (q!=p): (1 - p_end - p_sustain) / (N_pitches - 1) * geometric(|p-q|)

Backed by numpy log-domain Viterbi. Runs at ~10k frames/second on CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, hz_to_midi, midi_to_hz


@dataclass
class HMMConfig:
    midi_lo: int = 36       # C2
    midi_hi: int = 96       # C7
    p_sustain: float = 0.93
    p_end: float = 0.04
    p_start: float = 0.05
    sigma_voicing: float = 0.30
    sigma_midi: float = 0.50
    interval_decay: float = 0.50  # geometric over |p-q| semitones for jumps


def segment_pitch_to_notes_hmm(
    times: np.ndarray,
    hz: np.ndarray,
    voicing: np.ndarray,
    mc: ModeConfig,
    hmm: HMMConfig | None = None,
) -> list[NoteEvent]:
    if len(times) == 0:
        return []
    hmm = hmm or HMMConfig()
    midi_obs = np.where(hz > 0, np.array([hz_to_midi(float(h)) for h in hz]), 0.0)
    states = _states(hmm)
    n_states = len(states)
    log_em = _emission_log_probs(midi_obs, voicing, hmm)
    log_trans = _transition_log(hmm)

    n_frames = len(times)
    dp = np.full((n_frames, n_states), -1e18, dtype=np.float64)
    bk = np.full((n_frames, n_states), -1, dtype=np.int64)
    dp[0] = log_em[0]
    for t in range(1, n_frames):
        prev = dp[t - 1].reshape(-1, 1)
        scores = prev + log_trans
        best = np.argmax(scores, axis=0)
        dp[t] = scores[best, np.arange(n_states)] + log_em[t]
        bk[t] = best

    path = np.empty(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = int(bk[t + 1, path[t + 1]])

    return _path_to_notes(path, times, voicing, hmm, mc)


def _states(hmm: HMMConfig) -> np.ndarray:
    return np.concatenate([[-1], np.arange(hmm.midi_lo, hmm.midi_hi + 1, dtype=np.int64)])


def _emission_log_probs(midi_obs: np.ndarray, voicing: np.ndarray, hmm: HMMConfig) -> np.ndarray:
    n = len(midi_obs)
    pitches = np.arange(hmm.midi_lo, hmm.midi_hi + 1, dtype=np.float64)
    n_states = len(pitches) + 1
    out = np.empty((n, n_states), dtype=np.float64)
    sv2 = 2 * hmm.sigma_voicing * hmm.sigma_voicing
    sm2 = 2 * hmm.sigma_midi * hmm.sigma_midi
    out[:, 0] = -(voicing ** 2) / sv2
    diff_v = ((voicing - 1.0) ** 2) / sv2
    diff_m = ((midi_obs[:, None] - pitches[None, :]) ** 2) / sm2
    diff_m = np.where(midi_obs[:, None] > 0, diff_m, 4.0)
    out[:, 1:] = -(diff_v[:, None] + diff_m)
    return out


def _transition_log(hmm: HMMConfig) -> np.ndarray:
    pitches = np.arange(hmm.midi_lo, hmm.midi_hi + 1, dtype=np.int64)
    np_pitches = len(pitches)
    n = np_pitches + 1
    T = np.full((n, n), 1e-9, dtype=np.float64)
    T[0, 0] = 1.0 - hmm.p_start
    T[0, 1:] = hmm.p_start / np_pitches
    p_jump_total = max(1.0 - hmm.p_end - hmm.p_sustain, 0.0)
    intervals = np.abs(pitches[:, None] - pitches[None, :])
    weights = (hmm.interval_decay ** intervals).astype(np.float64)
    np.fill_diagonal(weights, 0.0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    T[1:, 0] = hmm.p_end
    T[1:, 1:] = p_jump_total * weights
    np.fill_diagonal(T[1:, 1:], hmm.p_sustain)
    T = T / T.sum(axis=1, keepdims=True)
    return np.log(T)


def _path_to_notes(path: np.ndarray, times: np.ndarray, voicing: np.ndarray,
                   hmm: HMMConfig, mc: ModeConfig) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    n = len(path)
    if n == 0:
        return notes
    pitches = np.concatenate([[-1], np.arange(hmm.midi_lo, hmm.midi_hi + 1, dtype=np.int64)])
    cur_state = path[0]
    seg_start = 0
    frame_step = float(times[1] - times[0]) if n > 1 else 0.01
    for t in range(1, n):
        if path[t] != cur_state:
            if cur_state != 0:
                notes.append(_make_note_from_seg(times, voicing, seg_start, t - 1, int(pitches[cur_state]), frame_step))
            cur_state = path[t]
            seg_start = t
    if cur_state != 0:
        notes.append(_make_note_from_seg(times, voicing, seg_start, n - 1, int(pitches[cur_state]), frame_step))
    notes = [n_ for n_ in notes if (n_.offset_s - n_.onset_s) >= mc.min_note_seconds]
    return notes


def _make_note_from_seg(times: np.ndarray, voicing: np.ndarray, s: int, e: int,
                        midi_pitch: int, frame_step: float) -> NoteEvent:
    return NoteEvent(
        onset_s=float(times[s]),
        offset_s=float(times[e]) + frame_step,
        pitch_midi=int(midi_pitch),
        pitch_hz=float(midi_to_hz(midi_pitch)),
        confidence=float(np.mean(voicing[s:e + 1])),
    )
