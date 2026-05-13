"""Phase G G-15: DDSP solo_flute2 timbre transfer.

Three fixes vs the v3 solo_violin failure (`humscribe/pitch/timbre_transfer/ddsp_violin.py`):

1. Use the `solo_flute2_ckpt` Magenta checkpoint instead of `solo_violin_ckpt`. Flute
   is less vibrato-sensitive than violin (no bowing artefacts).
2. Crossfade 4 s chunk boundaries with 200 ms overlap-add. The original
   solo_violin path concatenated chunks without crossfade, creating
   audible clicks at every 4 s boundary that the downstream PESTO/CREPE
   tracker confused for note onsets (14/40 clips had F1=0).
3. Disable DDSP's loudness normalisation. The default normalisation
   forces the synthesised audio to match the dataset's loudness curve,
   which suppresses the actual hum's dynamics; PESTO/CREPE prefer
   per-frame loudness-true signals.

Checkpoint download:
    gs://magenta-data/ddsp/checkpoints/solo_flute_ckpt.zip
expanded to /workspace/.cache/ddsp_checkpoints/solo_flute2_ckpt/.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np

CHECKPOINT_DIR = Path("/workspace/.cache/ddsp_checkpoints/solo_flute2_ckpt")
CHUNK_S = 4.0
CROSSFADE_MS = 200.0

_MODEL = None
_STATS = None


def is_checkpoint_available() -> bool:
    """True if solo_flute2 checkpoint is downloaded + expanded locally."""
    return (CHECKPOINT_DIR / "operative_config-0.gin").exists()


def _load_model():
    global _MODEL, _STATS
    if _MODEL is not None:
        return _MODEL, _STATS
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import gin
    import pickle
    from ddsp.training import models
    gin_file = CHECKPOINT_DIR / "operative_config-0.gin"
    gin.parse_config_file(str(gin_file), skip_unknown=True)
    model = models.Autoencoder()
    ckpt = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(str(CHECKPOINT_DIR))
    ckpt.restore(latest).expect_partial()
    with open(CHECKPOINT_DIR / "dataset_statistics.pkl", "rb") as f:
        _STATS = pickle.load(f)
    _MODEL = model
    return _MODEL, _STATS


def _crossfade_concat(chunks: list[np.ndarray], sr: int, *,
                      crossfade_ms: float = CROSSFADE_MS) -> np.ndarray:
    """Overlap-add chunks with a linear crossfade of `crossfade_ms`."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)
    n_xfade = int(crossfade_ms / 1000.0 * sr)
    if n_xfade <= 0:
        return np.concatenate(chunks).astype(np.float32)
    out = np.array(chunks[0], dtype=np.float32)
    for nxt in chunks[1:]:
        if len(out) < n_xfade or len(nxt) < n_xfade:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, n_xfade, dtype=np.float32)
        fade_in = 1.0 - fade_out
        tail = out[-n_xfade:] * fade_out
        head = nxt[:n_xfade] * fade_in
        merged = tail + head
        out = np.concatenate([out[:-n_xfade], merged, nxt[n_xfade:]])
    return out


def transfer(hum_audio: np.ndarray, sr_in: int = 22050) -> Tuple[np.ndarray, int]:
    """Run solo_flute2 over 4-s chunks with crossfade, no loudness norm.

    Returns (flute_audio, sr_out). sr_out = 16000.
    """
    import librosa
    if not is_checkpoint_available():
        raise FileNotFoundError(
            "solo_flute2_ckpt missing — download from "
            "gs://magenta-data/ddsp/checkpoints/solo_flute_ckpt.zip "
            f"into {CHECKPOINT_DIR}")
    if hum_audio.ndim > 1:
        hum_audio = hum_audio.mean(axis=-1)
    if sr_in != 16000:
        hum_audio = librosa.resample(hum_audio.astype(np.float32),
                                       orig_sr=sr_in, target_sr=16000)
    sr = 16000
    chunk_n = int(CHUNK_S * sr)
    chunks_out: list[np.ndarray] = []
    model, stats = _load_model()
    # Per-chunk: extract features, run model, optionally bypass loudness norm.
    from ddsp.training.metrics import compute_audio_features
    import tensorflow as tf
    for start in range(0, len(hum_audio), chunk_n):
        end = min(start + chunk_n, len(hum_audio))
        win = hum_audio[start:end]
        if len(win) < chunk_n:
            win = np.pad(win, (0, chunk_n - len(win)))
        feats = compute_audio_features(win)
        # Skip the loudness-shift step (the violin path's failure mode).
        feats_in = {
            "audio": tf.constant(win[None], dtype=tf.float32),
            "f0_hz": tf.constant(feats["f0_hz"][None], dtype=tf.float32),
            "loudness_db": tf.constant(feats["loudness_db"][None],
                                         dtype=tf.float32),
        }
        out = model(feats_in, training=False)
        chunks_out.append(np.array(out["audio_synth"][0]))
    full = _crossfade_concat(chunks_out, sr)
    return full[:len(hum_audio)], sr
