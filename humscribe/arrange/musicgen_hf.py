"""MusicGen-Melody arrangement using HuggingFace `transformers` (Phase E item 4).

This is the cross-platform replacement for `humscribe/arrange/musicgen.py`,
which depends on `audiocraft` (Linux/macOS-only, brittle deps). The HF
`MusicgenMelodyForConditionalGeneration` uses the same Meta weights but
is pure transformers + torch and works on Windows.

Public API is the same as `humscribe.arrange.musicgen.arrange` so callers can
switch backends with one import line.
"""
from __future__ import annotations
import io
import os
from typing import Literal

import numpy as np
import torch
import soundfile as _sf
import torchaudio.functional as _F  # type: ignore

# Model class names changed between transformers 4.45 → 4.50; we accept either.
try:
    from transformers import MusicgenMelodyForConditionalGeneration as _MusicgenMelody
    from transformers import AutoProcessor as _AutoProcessor
    _HF_OK = True
except ImportError:
    _HF_OK = False


_LOADED: dict[str, tuple] = {}

# Map our "melody" / "melody-large" labels to the HF repo IDs.
HF_REPO = {
    "melody": "facebook/musicgen-melody",
    "melody-large": "facebook/musicgen-melody-large",
}


PROMPT_PRESETS: dict[str, str] = {
    "lo-fi hip hop": "lo-fi hip hop with mellow piano, warm vinyl noise, soft drums",
    "jazz trio": "jazz trio with acoustic guitar, double bass, brushed drums",
    "EDM": "energetic EDM with synth bass, layered pads, four-on-the-floor kick",
    "orchestral cinematic": "orchestral cinematic strings with brass swells",
    "indie folk": "indie folk with acoustic guitar and brushed drums",
    "bossa nova": "bossa nova with nylon guitar, soft percussion, walking bass",
}


def _load(model_size: Literal["melody", "melody-large"] = "melody",
          device: str | None = None,
          dtype: torch.dtype = torch.float16,
          lora_adapter: str | None = None) -> tuple:
    """Return (model, processor) cached by (size, dtype, adapter)."""
    if not _HF_OK:
        raise RuntimeError(
            "transformers does not expose MusicgenMelodyForConditionalGeneration. "
            "Install transformers >= 4.45 to use the HF backend."
        )
    key = f"{model_size}:{dtype}:{lora_adapter or ''}"
    if key in _LOADED:
        return _LOADED[key]
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    repo = HF_REPO[model_size]
    model = _MusicgenMelody.from_pretrained(repo, torch_dtype=dtype).to(target)
    processor = _AutoProcessor.from_pretrained(repo)
    if lora_adapter is not None:
        # PEFT adapters trained on the audiocraft-style LM may need translation
        # to the HF state dict layout. We accept a directory and attempt to
        # load it as a PEFT adapter; if that fails, raise so the user knows
        # they need to retrain or use the audiocraft path.
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(lora_adapter))
            model.eval()
        except Exception as e:
            raise RuntimeError(
                f"could not load LoRA adapter from {lora_adapter!r} onto HF "
                f"MusicgenMelody: {e}. The B77 adapter was trained against "
                "audiocraft's LM and may need conversion."
            ) from e
    model.eval()
    _LOADED[key] = (model, processor)
    return model, processor


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
    """Generate a melody-conditioned arrangement via HF transformers.

    Returns 16 kHz mono WAV bytes (transformers' MusicgenMelody outputs at
    32 kHz the same as audiocraft; we downsample-encode but keep 32 kHz to
    match the audiocraft API).
    """
    if seed is not None:
        torch.manual_seed(seed)
    model, processor = _load(model_size=model_size, lora_adapter=lora_adapter)
    device = next(model.parameters()).device

    audio_np, sr = _sf.read(str(melody_audio_path), always_2d=True)
    melody = torch.from_numpy(audio_np.T.astype("float32"))  # (channels, T)
    if sr != 32000:
        melody = _F.resample(melody, sr, 32000)
    if melody.shape[0] > 1:
        melody = melody.mean(0, keepdim=True)
    if melody.shape[-1] < 32000:
        pad = torch.zeros(1, 32000 - melody.shape[-1])
        melody = torch.cat([melody, pad], dim=-1)
    melody_np = melody.squeeze(0).numpy()  # (T,)

    inputs = processor(
        audio=melody_np, sampling_rate=32000,
        text=[prompt], padding=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # MusicgenMelody yields 50 audio tokens/sec; total tokens ≈ duration*50.
    max_new_tokens = int(round(50 * duration_s))
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        guidance_scale=float(cfg_coef),
        temperature=float(temperature),
        top_k=250,
    )
    with torch.inference_mode():
        audio_tokens = model.generate(**inputs, **gen_kwargs)

    sample_rate = model.config.audio_encoder.sampling_rate  # 32000
    wav = audio_tokens[0, 0].cpu().numpy().astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767).astype(np.int16)
    buf = io.BytesIO()
    import scipy.io.wavfile as wavfile
    wavfile.write(buf, int(sample_rate), pcm16)
    return buf.getvalue()


def arrange_to_file(melody_audio_path: str, prompt: str, out_path: str, **kw) -> str:
    data = arrange(melody_audio_path, prompt, **kw)
    p = os.fspath(out_path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


def available() -> bool:
    """Return True when transformers exposes MusicgenMelody. Used by
    `humscribe.arrange.musicgen.choose_backend` to decide between the
    audiocraft path (default) and this one."""
    return _HF_OK
