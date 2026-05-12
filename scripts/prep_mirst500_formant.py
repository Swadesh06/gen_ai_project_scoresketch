"""Phase F-2b prep: extract formant-band mel-spectrogram from MIR-ST500 audio.

For each downloaded MIR-ST500 mp3 (in /workspace/datasets/mirst500/audio_partial/),
compute 80-bin mel-spectrogram restricted to 1500-3500 Hz at 10 ms hop,
22050 Hz audio. Saves to /workspace/.cache/mirst500_formant/<id>.npz.

These features are the F-2b pretraining input. Labels come from the
MIR-ST500_corrected.json file (onset, offset, pitch per song).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import librosa

AUDIO_DIR = Path("/workspace/datasets/mirst500/audio_partial")
OUT = Path("/workspace/.cache/mirst500_formant")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    mp3s = sorted(AUDIO_DIR.glob("*.mp3"), key=lambda p: int(p.stem))
    print(f"found {len(mp3s)} MIR-ST500 mp3s")
    n_done = n_skip = 0
    for mp3 in mp3s:
        out_path = OUT / f"{mp3.stem}.npz"
        if out_path.exists():
            n_done += 1; continue
        try:
            y, sr = librosa.load(str(mp3), sr=22050)
        except Exception as e:
            print(f"  load fail {mp3.stem}: {e}"); n_skip += 1; continue
        # Trim to first 60 seconds (training is per-30s window in F-2)
        y = y[: 60 * sr]
        try:
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=80,
                fmin=1500.0, fmax=3500.0,
                hop_length=220, n_fft=2048,
            )
            log_mel = librosa.power_to_db(mel)
            log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / (log_mel.std(axis=1, keepdims=True) + 1e-3)
            np.savez(str(out_path), mel=log_mel.astype(np.float16),
                      duration_s=float(len(y) / sr))
            n_done += 1
        except Exception as e:
            print(f"  mel fail {mp3.stem}: {e}"); n_skip += 1
        if n_done % 10 == 0:
            print(f"  [{n_done}/{len(mp3s)}] processed")
    print(f"DONE: {n_done} processed, {n_skip} skipped -> {OUT}")


if __name__ == "__main__":
    main()
