"""Phase F-2e formant offset corrector (production wiring).

After the heuristic segmenter places a note offset, this module snaps the
offset to the nearest BiLSTM peak within a small search window — but only
when the BiLSTM is confident. Lifts Vocadito offset20 F1 by +0.027 over
the heuristic baseline (0.343 → 0.370).

Default knobs (tuned via scripts/eval_f2e_threshold_sweep.py):
- search_ms = 50    # symmetric window around heuristic offset
- min_prob  = 0.30  # BiLSTM probability floor for accepting the snap
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np

from humscribe.notes import NoteEvent

_DEFAULT_CKPT = Path("checkpoints/formant_offset_vocadito/fold0.pt")
_DEFAULT_SR = 22050
_DEFAULT_HOP = 220  # 10 ms at 22050 Hz
_DEFAULT_N_FFT = 2048
_DEFAULT_F_MIN = 1500.0
_DEFAULT_F_MAX = 3500.0
_DEFAULT_N_MELS = 80


_MODEL_CACHE: dict[str, object] = {}


def _get_model(ckpt_path: Path = _DEFAULT_CKPT):
    """Cached model loader. First call ~50 ms; later calls reuse."""
    key = str(ckpt_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not ckpt_path.exists():
        return None
    import torch
    from humscribe.train.formant_offset import (
        FormantOffsetBiLSTM, FormantOffsetConfig,
    )
    cfg = FormantOffsetConfig(in_dim=80, hidden=96, layers=2, dropout=0.2)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = FormantOffsetBiLSTM(cfg)
    model.load_state_dict(state["model_state"])
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _audio_to_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """80-bin mel, 1500-3500 Hz, 10 ms hop. Returns (80, T) log-mel."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=_DEFAULT_N_MELS,
        fmin=_DEFAULT_F_MIN, fmax=_DEFAULT_F_MAX,
        hop_length=_DEFAULT_HOP, n_fft=_DEFAULT_N_FFT,
    )
    log_mel = librosa.power_to_db(mel)
    log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / (log_mel.std(axis=1, keepdims=True) + 1e-3)
    return log_mel


def _bilstm_probs(model, mel: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        x = torch.from_numpy(mel.T.astype(np.float32)).unsqueeze(0)
        logits = model(x).squeeze(0).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def correct_offsets(
    notes: list[NoteEvent],
    audio: np.ndarray,
    sr: int,
    *,
    min_prob: float = 0.30,
    search_ms: float = 50.0,
    ckpt_path: Path = _DEFAULT_CKPT,
) -> list[NoteEvent]:
    """Apply the F-2e confidence-head correction to a list of NoteEvents.

    Returns a new list of NoteEvents with offsets snapped to BiLSTM peaks
    where the model is confident. Onsets and pitches are unchanged.

    If the BiLSTM checkpoint is missing or any step fails, the input
    notes are returned unchanged (safe-by-default).
    """
    if not notes:
        return notes
    model = _get_model(ckpt_path)
    if model is None:
        return notes
    try:
        mel = _audio_to_mel(audio, sr)
        probs = _bilstm_probs(model, mel)
    except Exception:
        return notes  # safe-by-default on any extraction failure

    hop_s = _DEFAULT_HOP / float(sr)
    window = int(search_ms / 1000.0 / hop_s)
    n_frames = len(probs)

    out: list[NoteEvent] = []
    for n in notes:
        center = int(n.offset_s / hop_s)
        lo = max(0, center - window)
        hi = min(n_frames, center + window + 1)
        if lo >= hi:
            out.append(n); continue
        sub = probs[lo:hi]
        idx = int(np.argmax(sub))
        if sub[idx] < min_prob:
            out.append(n); continue
        new_off = (lo + idx) * hop_s
        # Sanity: keep min duration 50ms
        if new_off - n.onset_s < 0.05:
            new_off = n.onset_s + 0.05
        new = NoteEvent(
            onset_s=n.onset_s,
            offset_s=new_off,
            pitch_hz=n.pitch_hz,
            pitch_midi=n.pitch_midi,
            velocity=n.velocity,
            confidence=n.confidence,
        )
        out.append(new)
    return out
