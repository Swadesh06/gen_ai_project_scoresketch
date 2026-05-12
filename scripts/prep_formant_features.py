"""Phase F-2 prep: formant-band mel-spectrograms for Vocadito.

For each Vocadito clip, compute an 80-bin mel-spectrogram restricted to
1500-3500 Hz (the vocal formant band), 10 ms hop, 22050 Hz audio. Cache
to /workspace/.cache/voc_formant/<clip>.npz alongside the existing
voc_*.npz caches.

This data is for the F-2 formant-band offset detector (not yet built).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VOC_AUDIO = Path("/home/swadesh/datasets/vocadito/Audio")
OUT = Path("/workspace/.cache/voc_formant")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    wavs = sorted(VOC_AUDIO.glob("vocadito_*.wav"),
                  key=lambda p: int(p.stem.split("_")[1]))
    for i, wav in enumerate(wavs):
        out_path = OUT / f"{wav.stem}.npz"
        if out_path.exists():
            print(f"have {wav.stem}")
            continue
        y, sr = librosa.load(str(wav), sr=22050)
        # 80-bin mel, but restricted to 1500-3500 Hz formant band.
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=80,
            fmin=1500.0, fmax=3500.0,
            hop_length=220,  # 10 ms at 22050
            n_fft=2048,
        )
        log_mel = librosa.power_to_db(mel)
        # Per-clip per-bin normalization to zero mean/unit variance.
        log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / (log_mel.std(axis=1, keepdims=True) + 1e-3)
        np.savez(str(out_path), mel=log_mel.astype(np.float16),
                  duration_s=float(len(y) / sr))
        if i % 10 == 0:
            print(f"[{i}] {wav.stem}  shape={log_mel.shape}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
