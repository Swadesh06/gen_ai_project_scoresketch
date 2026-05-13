"""DDSP solo-violin timbre transfer wrapper.

Magenta solo_violin_ckpt (ckpt-40000, ~58 MB) lives at
/workspace/.cache/ddsp_checkpoints/solo_violin_ckpt/. The autoencoder
architecture is fully described by `operative_config-0.gin` in that dir.

Usage:
    from humscribe.pitch.timbre_transfer.ddsp_violin import transfer
    violin_audio, sr_out = transfer(hum_audio, sr_in=22050)
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = Path("/workspace/.cache/ddsp_checkpoints/solo_violin_ckpt")

_MODEL = None
_STATS = None


def _load_model():
    global _MODEL, _STATS
    if _MODEL is not None:
        return _MODEL, _STATS
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
    if latest is None:
        latest = str(CHECKPOINT_DIR / "ckpt-40000")
    ckpt.restore(latest).expect_partial()
    with open(CHECKPOINT_DIR / "dataset_statistics.pkl", "rb") as f:
        _STATS = pickle.load(f)
    _MODEL = model
    return _MODEL, _STATS


def _features(audio: np.ndarray, sr: int) -> tuple[dict, np.ndarray]:
    """Extract f0, loudness, audio at 16 kHz for DDSP autoencoder."""
    from ddsp.training.metrics import compute_audio_features
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32),
                                 orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    # 4-second window at 16 kHz (the canonical DDSP demo length).
    n_target = 4 * sr
    if len(audio) < n_target:
        audio = np.pad(audio, (0, n_target - len(audio)))
    audio = audio[:n_target]
    feats = compute_audio_features(audio)
    feats = dict(feats)
    feats["audio"] = audio
    feats = {k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) and v.ndim == 1 else (
                 np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) and v.ndim == 0 else v)
             for k, v in feats.items()}
    return feats, audio


def _normalize_features(feats: dict, stats: dict) -> dict:
    if "mean_loudness_db" in stats:
        cur = float(np.mean(feats["loudness_db"]))
        feats = dict(feats)
        feats["loudness_db"] = feats["loudness_db"] + (stats["mean_loudness_db"] - cur)
    return feats


def transfer_4s(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Convert a single ≤ 4 s hummed audio clip to violin-timbred audio.

    Returns (violin_audio, 16000).
    """
    model, stats = _load_model()
    feats, src_audio = _features(audio, sr)
    feats = _normalize_features(feats, stats)
    out = model(feats, training=False)
    audio_out = model.get_audio_from_outputs(out)
    audio_out = np.asarray(audio_out)
    if audio_out.ndim > 1:
        audio_out = audio_out.squeeze(0)
    return audio_out, 16000


def transfer(audio: np.ndarray, sr: int, chunk_s: float = 4.0,
              overlap_s: float = 0.0) -> tuple[np.ndarray, int]:
    """Tile DDSP transfer across an arbitrarily long clip.

    Splits the audio into chunk_s-second windows, transfers each, and
    concatenates the outputs. Output sample rate is 16000.
    """
    import librosa
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32),
                                 orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    chunk_n = int(chunk_s * sr)
    if len(audio) <= chunk_n:
        return transfer_4s(audio, sr)
    pieces = []
    for start in range(0, len(audio), chunk_n):
        chunk = audio[start:start + chunk_n]
        if len(chunk) < int(0.5 * sr):
            break  # skip <0.5s tail
        out, _ = transfer_4s(chunk, sr)
        pieces.append(out)
    return np.concatenate(pieces), sr
