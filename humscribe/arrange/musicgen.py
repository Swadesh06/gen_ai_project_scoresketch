"""MusicGen-Melody arrangement post-stage (B+2 work item 3).

Source: https://huggingface.co/facebook/musicgen-melody-large
License: code MIT, weights CC-BY-NC-4.0 (document in README)

Loads either musicgen-melody (1.5B, ~5 GB VRAM) or musicgen-melody-large
(3.3B, ~13 GB VRAM at fp16). Default is the smaller variant for the live
demo flow; pass `model_size="melody-large"` for higher-quality offline use.

Stubs `spacy/thinc` because `audiocraft.modules.conditioners` imports them at
package-import time even though they aren't reached for chroma/text inference.
"""
from __future__ import annotations
import io
import os
import sys
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

# audiocraft hard-imports spacy in modules/conditioners.py; we don't use the
# CLAP/Roberta paths so a stub is enough.
class _Stub:
    def __getattr__(self, _n):  # pragma: no cover
        return _Stub()
    def __call__(self, *_a, **_k):  # pragma: no cover
        return _Stub()
for _m in ("spacy", "spacy.lang.en", "thinc", "thinc.util"):
    sys.modules.setdefault(_m, _Stub())

import soundfile as _sf
import torchaudio.functional as _F  # type: ignore
from audiocraft.models import MusicGen


_LOADED: dict[str, MusicGen] = {}


def _load(model_size: Literal["melody", "melody-large"] = "melody",
          device: str | None = None,
          dtype: torch.dtype = torch.float16,
          lora_adapter: str | None = None) -> MusicGen:
    """Cache-once loader; subsequent calls reuse the model.

    If `lora_adapter` is set (path to a PEFT adapter checkpoint, e.g.
    `checkpoints/musicgen_lora_b77/step_300`), the adapter is loaded over
    the LM. Cached separately per (model_size, dtype, adapter) tuple.
    """
    key = f"{model_size}:{dtype}:{lora_adapter or ''}"
    if key in _LOADED:
        return _LOADED[key]
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = f"facebook/musicgen-{model_size}"
    model = MusicGen.get_pretrained(name, device=target)
    if lora_adapter is not None:
        # PEFT LoRA was trained against fp32 LM (per B74/B77). Cast to fp32
        # before attaching adapters to keep dtypes consistent.
        from peft import PeftModel
        model.lm = model.lm.to(torch.float32)
        model.lm = PeftModel.from_pretrained(model.lm, str(lora_adapter))
        model.lm.eval()
    elif dtype == torch.float16 and target == "cuda":
        for p in model.lm.parameters():
            p.data = p.data.to(dtype)
    _LOADED[key] = model
    return model


PROMPT_PRESETS: dict[str, str] = {
    "lo-fi hip hop": "lo-fi hip hop with mellow piano, warm vinyl noise, soft drums",
    "jazz trio": "jazz trio with acoustic guitar, double bass, brushed drums",
    "EDM": "energetic EDM with synth bass, layered pads, four-on-the-floor kick",
    "orchestral cinematic": "orchestral cinematic strings with brass swells",
    "indie folk": "indie folk with acoustic guitar and brushed drums",
    "bossa nova": "bossa nova with nylon guitar, soft percussion, walking bass",
}


def arrange(
    melody_audio_path: str,
    prompt: str,
    duration_s: float = 15.0,
    model_size: Literal["melody", "melody-large"] = "melody",
    seed: int | None = 0,
    cfg_coef: float = 3.0,
    temperature: float = 1.0,
    lora_adapter: str | None = None,
) -> bytes:
    """Generate a melody-conditioned arrangement.

    Returns 32 kHz stereo (or mono on smaller variants) WAV bytes ready for
    `st.audio` / file write.

    If `lora_adapter` is given (path to a PEFT checkpoint, e.g.
    `checkpoints/musicgen_lora_b77/step_300`), the LM uses that adapter on
    top of the base weights — fine-tuned style/speaker.
    """
    if seed is not None:
        torch.manual_seed(seed)
    mg = _load(model_size=model_size, lora_adapter=lora_adapter)
    # Bypass torchaudio.load (which now requires torchcodec/ffmpeg). soundfile
    # is already a pipeline dependency.
    audio_np, sr = _sf.read(str(melody_audio_path), always_2d=True)
    melody = torch.from_numpy(audio_np.T.astype("float32"))  # (channels, T)
    if sr != 32000:
        melody = _F.resample(melody, sr, 32000)
    if melody.shape[0] > 1:
        melody = melody.mean(0, keepdim=True)
    if melody.shape[-1] < 32000:  # too short
        pad = torch.zeros(1, 32000 - melody.shape[-1])
        melody = torch.cat([melody, pad], dim=-1)
    melody_batch = melody.unsqueeze(0)  # (1, 1, T)

    mg.set_generation_params(
        duration=float(duration_s), top_k=250, top_p=0.0,
        temperature=float(temperature), cfg_coef=float(cfg_coef),
    )
    with torch.inference_mode():
        wav = mg.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=melody_batch,
            melody_sample_rate=32000,
        )
    audio = wav[0].cpu().numpy()
    if audio.ndim == 1:
        audio = audio[None, :]
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio.T * 32767).astype(np.int16)
    buf = io.BytesIO()
    import scipy.io.wavfile as wavfile
    wavfile.write(buf, 32000, pcm16)
    return buf.getvalue()


def arrange_to_file(melody_audio_path: str, prompt: str, out_path: str,
                    **kw) -> str:
    """Convenience: write the arrangement WAV directly to disk."""
    data = arrange(melody_audio_path, prompt, **kw)
    p = os.fspath(out_path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p
