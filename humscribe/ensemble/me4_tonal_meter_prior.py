"""ME-4 — tonal-meter prior on the DP.

Music theory: in tonal music, strong scale degrees (tonic 1, dominant 5,
subdominant 4) preferentially land on metrically strong positions (beats 1
and 3 in 4/4, beat 1 in 3/4). Weak degrees (#4, b6, ...) are more likely on
off-beats. A DP that has no metric knowledge picks tatum positions purely
on note-time-vs-beat-time distance — it has no way to break ties when a
note is equidistant between two candidate tatums.

This member produces a per-(tatum_position, scale_degree) bonus that the
DP adds as a tie-breaker. The prior is empirical, derived from a small
corpus of Bach chorales (music21 ships them locally).

Strategy:
1. Walk every Bach four-part chorale in music21's corpus.
2. For every note, compute its (beat_position_in_bar, scale_degree).
3. Accumulate P(degree | beat_position) probability tables.
4. At inference, key estimation gives us the tonic; we look up the
   note's scale degree and add log P(degree | candidate_tatum_position)
   as a soft prior to the DP's score.

Result is cached in `.cache/me4_tonal_prior.npz` so we don't re-compute
on every call.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

CACHE = Path("/workspace/.cache/me4_tonal_prior.npz")


def _scale_degree(midi: int, tonic_pc: int, is_major: bool) -> int:
    """Return scale degree 0..11 where 0 = tonic. Negative if pitch is None."""
    if midi < 1: return -1
    return (int(midi) - int(tonic_pc)) % 12


def build_prior_from_music21_corpus(max_pieces: int = 200) -> dict:
    """Walk Bach chorales and accumulate degree-by-beat probabilities."""
    from music21.corpus.chorales import Iterator
    from music21 import key as m21key, note as m21note, chord as m21chord, meter as m21meter
    grid = np.zeros((12, 12), dtype=np.float64)  # (beat_position_in_bar_quarter, scale_degree)
    n_seen = 0
    for i, score in enumerate(Iterator()):
        if i >= max_pieces:
            break
        try:
            k_sol = score.analyze("key")
            tonic_pc = int(k_sol.tonic.midi) % 12
            is_major = k_sol.mode == "major"
        except Exception:
            continue
        # Use top-level offset modulo bar length.
        ts_list = list(score.recurse().getElementsByClass(m21meter.TimeSignature))
        if not ts_list:
            ts_num = 4; ts_den = 4
        else:
            ts_num = ts_list[0].numerator; ts_den = ts_list[0].denominator
        bar_len_ql = ts_num * 4.0 / ts_den
        if bar_len_ql <= 0: continue
        for el in score.recurse().notes:
            offs = float(el.offset)
            beat_in_bar = offs % bar_len_ql
            beat_idx = int(np.floor(beat_in_bar * 12 / max(bar_len_ql, 0.001)))
            beat_idx = max(0, min(11, beat_idx))
            if isinstance(el, m21note.Note):
                pcs = [int(el.pitch.midi) % 12]
            elif isinstance(el, m21chord.Chord):
                pcs = [int(p.midi) % 12 for p in el.pitches]
            else:
                pcs = []
            for pc in pcs:
                degree = (pc - tonic_pc) % 12
                grid[beat_idx, degree] += 1.0
            n_seen += 1
        if i % 20 == 0:
            print(f"  ME-4 corpus prior: {i+1} pieces, {n_seen} notes")
    # Normalise each row to a probability.
    row_sums = grid.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = grid / row_sums
    return {"grid": grid, "probs": probs, "n_pieces": i + 1, "n_notes": n_seen}


def load_or_build_prior() -> dict:
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        return {"grid": d["grid"], "probs": d["probs"]}
    print("building ME-4 tonal-meter prior from music21 Bach corpus...")
    res = build_prior_from_music21_corpus(max_pieces=200)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(CACHE), grid=res["grid"], probs=res["probs"])
    print(f"  saved to {CACHE}: {res['n_notes']} notes")
    return res


def tonal_meter_log_prior(midi: int, tatum_pos: int, tatums_per_beat: int,
                          beats_per_bar: int, tonic_pc: int,
                          probs: np.ndarray) -> float:
    """Return log P(degree | beat-position-in-bar) for ME-4's DP tie-break."""
    if midi < 1: return 0.0
    degree = (int(midi) - int(tonic_pc)) % 12
    # tatum_pos is in tatums; map to 0..11.
    tatums_per_bar = max(1, tatums_per_beat * beats_per_bar)
    beat_idx = int(round((tatum_pos % tatums_per_bar) * 12 / tatums_per_bar))
    beat_idx = max(0, min(11, beat_idx))
    p = float(probs[beat_idx, degree])
    # Avoid log(0) by floor.
    return float(np.log(max(p, 1e-6)))
