"""Cemgil-Kappen DP rhythm quantization.

Onsets are quantized via Viterbi DP (Gaussian observation cost + off-grid
transition penalty). Offsets are quantized by snapping the observed duration to
the nearest *musically allowed* tatum count — which dramatically reduces the
"7/12 quarter" type artefacts that pure rounding produces (Phase B Exp B1).
"""
from __future__ import annotations
import numpy as np

# Allowed durations in tatums for common rhythmic values. Built per-TPB so 32nd
# notes are exactly representable when TPB allows it.
DEFAULT_ALLOWED_DURATIONS_TATUMS_TPB12 = np.array(
    [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48], dtype=np.int64,
)
DEFAULT_ALLOWED_DURATIONS_TATUMS_TPB24 = np.array(
    # 32nd, 16th-trip, 16th, dot-16th, 8th-trip, 8th, dot-8th, q-trip, quarter,
    # dot-quarter, half-trip, half, dot-half, whole, etc.
    [3, 4, 6, 9, 8, 12, 18, 16, 24, 36, 32, 48, 72, 96], dtype=np.int64,
)


def default_allowed_durations(tatums_per_beat: int) -> np.ndarray | None:
    if tatums_per_beat == 12:
        return DEFAULT_ALLOWED_DURATIONS_TATUMS_TPB12
    if tatums_per_beat == 24:
        return DEFAULT_ALLOWED_DURATIONS_TATUMS_TPB24
    return None


def adaptive_tatums_per_beat(beats: np.ndarray, slow_bpm_threshold: float = 70.0) -> int:
    if len(beats) < 2:
        return 12
    iois = np.diff(beats)
    iois = iois[(iois > 0.05) & (iois < 5.0)]
    if len(iois) == 0:
        return 12
    bpm = 60.0 / float(np.median(iois))
    return 24 if bpm < slow_bpm_threshold else 12


def viterbi_quantize_rhythm(
    onsets: np.ndarray,
    offsets: np.ndarray,
    beats: np.ndarray,
    tatums_per_beat: int = 12,
    sigma_tatums: float = 1.0,
    offgrid_penalty: float = 1.0,
    search_window_tatums: int = 6,
    allowed_durations_tatums: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    onsets = np.asarray(onsets, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    beats = np.asarray(beats, dtype=np.float64)
    n = len(onsets)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if len(beats) < 2:
        return np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)

    on_tatum_f = _to_tatum_floats(onsets, beats, tatums_per_beat)
    off_tatum_f = _to_tatum_floats(offsets, beats, tatums_per_beat)

    cands = _candidate_states(on_tatum_f, search_window_tatums)
    n_states = max(len(c) for c in cands)
    inf = np.float64(1e18)
    dp = np.full((n, n_states), inf)
    bk = np.full((n, n_states), -1, dtype=np.int64)

    obs0 = _gaussian_costs(on_tatum_f[0], cands[0], sigma_tatums)
    dp[0, :len(cands[0])] = obs0

    for i in range(1, n):
        prev = cands[i - 1]
        cur = cands[i]
        obs = _gaussian_costs(on_tatum_f[i], cur, sigma_tatums)
        for ci, ct in enumerate(cur):
            best = inf
            best_j = -1
            for pj, pt in enumerate(prev):
                if dp[i - 1, pj] >= inf:
                    continue
                if ct < pt:
                    trans = offgrid_penalty * 5.0
                else:
                    delta = ct - pt
                    frac = (delta % 1.0)
                    trans = offgrid_penalty * (1.0 - np.cos(np.pi * frac))
                total = dp[i - 1, pj] + obs[ci] + trans
                if total < best:
                    best = total
                    best_j = pj
            dp[i, ci] = best
            bk[i, ci] = best_j

    last_row = dp[n - 1, :len(cands[n - 1])]
    end_idx = int(np.argmin(last_row))
    q_on = np.zeros(n, dtype=np.int64)
    q_on[n - 1] = int(cands[n - 1][end_idx])
    cur_idx = end_idx
    for i in range(n - 1, 0, -1):
        prev_idx = bk[i, cur_idx]
        if prev_idx < 0:
            prev_idx = int(np.argmin(_gaussian_costs(on_tatum_f[i - 1], cands[i - 1], sigma_tatums)))
        q_on[i - 1] = int(cands[i - 1][prev_idx])
        cur_idx = prev_idx

    if allowed_durations_tatums is None:
        allowed_durations_tatums = default_allowed_durations(tatums_per_beat)
    q_off = _quantize_offsets(off_tatum_f, q_on, allowed_durations_tatums)
    return q_on, q_off


def _to_tatum_floats(times_s: np.ndarray, beats: np.ndarray, tatums_per_beat: int) -> np.ndarray:
    out = np.empty(len(times_s), dtype=np.float64)
    nb = len(beats)
    for i, t in enumerate(times_s):
        if t <= beats[0]:
            ioi = beats[1] - beats[0] if nb >= 2 else 0.5
            out[i] = (t - beats[0]) / ioi * tatums_per_beat
            continue
        if t >= beats[-1]:
            ioi = beats[-1] - beats[-2] if nb >= 2 else 0.5
            out[i] = (nb - 1) * tatums_per_beat + (t - beats[-1]) / ioi * tatums_per_beat
            continue
        j = int(np.searchsorted(beats, t)) - 1
        j = max(min(j, nb - 2), 0)
        ioi = beats[j + 1] - beats[j]
        if ioi <= 0:
            out[i] = j * tatums_per_beat
        else:
            out[i] = (j + (t - beats[j]) / ioi) * tatums_per_beat
    return out


def _candidate_states(on_tatum_f: np.ndarray, window: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for v in on_tatum_f:
        center = int(round(float(v)))
        lo = max(center - window, 0)
        hi = max(center + window + 1, lo + 1)
        out.append(np.arange(lo, hi, dtype=np.int64))
    return out


def _gaussian_costs(observed: float, candidates: np.ndarray, sigma: float) -> np.ndarray:
    diff = (candidates.astype(np.float64) - float(observed)) / max(float(sigma), 1e-6)
    return 0.5 * diff * diff


def _quantize_offsets(
    off_tatum_f: np.ndarray,
    q_on: np.ndarray,
    allowed_durations_tatums: np.ndarray | None,
) -> np.ndarray:
    q_off = np.empty(len(off_tatum_f), dtype=np.int64)
    if allowed_durations_tatums is None:
        for i, v in enumerate(off_tatum_f):
            rounded = int(round(float(v)))
            q_off[i] = max(rounded, q_on[i] + 1)
        return q_off
    allowed = np.sort(np.asarray(allowed_durations_tatums, dtype=np.int64))
    for i, v in enumerate(off_tatum_f):
        observed_dur = float(v) - float(q_on[i])
        if observed_dur <= 0:
            q_off[i] = q_on[i] + int(allowed[0])
            continue
        idx = int(np.argmin(np.abs(allowed.astype(np.float64) - observed_dur)))
        q_off[i] = q_on[i] + int(allowed[idx])
    return q_off
