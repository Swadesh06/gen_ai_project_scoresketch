"""Audio loading. Mono, float32, optional resample. ASAP MIDI fallback via FluidSynth."""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import soundfile as sf

DEFAULT_SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"


def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    p = Path(path).expanduser()
    if p.suffix.lower() in {".mid", ".midi"}:
        return _render_midi(p, target_sr or 22050)
    audio, sr = sf.read(str(p), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if target_sr is not None and sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    import librosa
    return librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)


def _render_midi(midi_path: Path, sr: int, sf2: str = DEFAULT_SF2) -> tuple[np.ndarray, int]:
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth not found; install fluid-soundfont-gm + fluidsynth to render MIDI")
    if not Path(sf2).exists():
        raise FileNotFoundError(f"SoundFont missing at {sf2}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    cmd = [
        "fluidsynth", "-ni", "-r", str(sr), "-F", wav_path, "-T", "wav",
        sf2, str(midi_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    audio, out_sr = sf.read(wav_path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), out_sr
