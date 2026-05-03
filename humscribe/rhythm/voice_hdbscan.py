"""HDBSCAN-based voice tracker (Phase B+1 Exp B48).

Density-based clustering of ByteDance notes by (time, pitch) features. Each
cluster becomes a 'voice' for the per-voice DP. Tackles the non-Bach failure
mode where greedy assignment over-fragments Romantic chordal textures.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import (
    default_allowed_durations, viterbi_quantize_rhythm,
)
from humscribe.rhythm.voice_tracking import per_voice_durations


@dataclass
class HDBSCANVoiceConfig:
    min_cluster_size: int = 4
    min_samples: int = 2
    pitch_weight: float = 0.6   # how much to weight pitch vs time in clustering
    time_window_s: float = 4.0  # local window for clustering


def assign_voices_hdbscan(notes: Sequence[NoteEvent], cfg: HDBSCANVoiceConfig | None = None) -> list[list[int]]:
    """Cluster notes by (time, pitch) into voices using HDBSCAN.
    Falls back gracefully to single voice if hdbscan not available."""
    cfg = cfg or HDBSCANVoiceConfig()
    n = len(notes)
    if n < cfg.min_cluster_size:
        return [list(range(n))]
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        return [list(range(n))]

    onsets = np.array([nv.onset_s for nv in notes], dtype=np.float64)
    midis = np.array([nv.midi() for nv in notes], dtype=np.float64)
    # Normalize: time uses local-window scale; pitch uses semitone scale
    t_norm = onsets / max(cfg.time_window_s, 1e-3)
    p_norm = midis * cfg.pitch_weight
    X = np.stack([t_norm, p_norm], axis=1)
    clusterer = HDBSCAN(min_cluster_size=cfg.min_cluster_size,
                        min_samples=cfg.min_samples, metric="euclidean")
    labels = clusterer.fit_predict(X)
    # Group indices by label; -1 (noise) goes to its own one-note voices
    voices: list[list[int]] = []
    label_to_voice: dict[int, int] = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            voices.append([i])
        else:
            if lab not in label_to_voice:
                label_to_voice[lab] = len(voices)
                voices.append([])
            voices[label_to_voice[lab]].append(i)
    return voices


def quantize_with_hdbscan_voice_tracking(
    notes: Sequence[NoteEvent],
    beats: np.ndarray,
    tatums_per_beat: int = 24,
    voice_cfg: HDBSCANVoiceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    voices = assign_voices_hdbscan(notes, voice_cfg)
    onsets, adj_offsets = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(
        onsets, adj_offsets, beats, tatums_per_beat=tatums_per_beat,
        allowed_durations_tatums=default_allowed_durations(tatums_per_beat),
    )
