"""Curate the app/demos/ set: 2 piano + 3 humming.

Picks short, expressive clips that exercise different modes well, clips
each one to ~25 seconds (so a full transcription fits in a Streamlit
spinner), and writes a small JSON metadata file that the UI uses to
auto-fill input_kind, suggested BPM, and suggested key for hard mode.

The five clips this writes:
  demo_1_piano_bach_bwv854.wav     piano  | clean fugue at C major / 4/4
  demo_2_piano_maestro_chamber.wav piano  | mixed acoustic, key/meter uncertain
  demo_3_humming_vocadito_1.wav    hum    | the "headline" Vocadito clip
  demo_4_humming_vocadito_8.wav    hum    | different singer, short
  demo_5_humming_vocadito_15.wav   hum    | mid-tempo, clean voicing
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parents[1]
DST = REPO / "app" / "demos"
TARGET_SR = 22050
CLIP_S = 25.0


def _resample_and_clip(src: Path, dst: Path, max_s: float = CLIP_S) -> tuple[int, float]:
    audio, sr = sf.read(str(src))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # Linear resample is fine for a presentation demo; quality is
        # bounded by the pipeline's downstream feature extractors anyway.
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, TARGET_SR)
        up, dn = TARGET_SR // g, sr // g
        audio = resample_poly(audio, up, dn).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    n_max = int(max_s * TARGET_SR)
    audio = audio[:n_max]
    # Peak normalise to -1 dBFS so the inline Streamlit st.audio() player
    # sounds clean on demo speakers.
    peak = float(np.max(np.abs(audio))) if len(audio) else 1.0
    if peak > 0:
        audio = audio * (10 ** (-1.0 / 20.0) / peak)
    sf.write(str(dst), audio, TARGET_SR)
    return TARGET_SR, len(audio) / TARGET_SR


SOURCES = [
    (
        "demo_1_piano_bach_bwv854.wav",
        Path("/workspace/.cache/asap_renders/Bach__Fugue__bwv_854.wav"),
        {
            "kind": "piano",
            "mode_default": "hard",
            "pitch_model": "pesto",
            "bpm_hint": 120,
            "key": "C major",
            "time_sig": "4/4",
            "label": "Piano - Bach Fugue BWV 854",
            "blurb": "ASAP-rendered Bach fugue. Use 'hard' mode to lock the key + time sig.",
        },
    ),
    (
        "demo_2_piano_maestro_chamber.wav",
        REPO / "outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--2_30s.wav",
        {
            "kind": "piano",
            "mode_default": "medium",
            "pitch_model": "pesto",
            "bpm_hint": 73,
            "key": None,
            "time_sig": None,
            "label": "Piano - MAESTRO chamber excerpt",
            "blurb": "Real recording; 'medium' mode with the BPM hint produces the cleanest score.",
        },
    ),
    (
        "demo_3_humming_vocadito_1.wav",
        Path("/workspace/.cache/vocadito_orig/Audio/vocadito_1.wav"),
        {
            "kind": "humming",
            "mode_default": "soft",
            "pitch_model": "pesto_crepevoicing",
            "bpm_hint": 115,
            "key": None,
            "time_sig": None,
            "label": "Humming - Vocadito clip 1 (headline G-4 example)",
            "blurb": "The Vocadito clip from the G-4 before/after figure. 'soft' mode is enough.",
        },
    ),
    (
        "demo_4_humming_vocadito_8.wav",
        Path("/workspace/.cache/vocadito_orig/Audio/vocadito_8.wav"),
        {
            "kind": "humming",
            "mode_default": "soft",
            "pitch_model": "pesto_crepevoicing",
            "bpm_hint": 95,
            "key": None,
            "time_sig": None,
            "label": "Humming - Vocadito clip 8 (different singer)",
            "blurb": "Variety; same humming branch handles a different vocal timbre.",
        },
    ),
    (
        "demo_5_humming_vocadito_15.wav",
        Path("/workspace/.cache/vocadito_orig/Audio/vocadito_15.wav"),
        {
            "kind": "humming",
            "mode_default": "medium",
            "pitch_model": "pesto_crepevoicing",
            "bpm_hint": 100,
            "key": None,
            "time_sig": None,
            "label": "Humming - Vocadito clip 15 (mid-tempo)",
            "blurb": "Tighter rhythmic figure; 'medium' mode with the BPM hint tightens snap.",
        },
    ),
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    # Move the prior demos to a backup folder so we don't lose them.
    backup = DST / "_prior_demos_backup"
    backup.mkdir(exist_ok=True)
    for f in DST.glob("demo_*.wav"):
        if f.name.startswith("demo_"):
            shutil.move(str(f), str(backup / f.name))
            print(f"backup  {f.name}")
    meta: list[dict] = []
    for fname, src, info in SOURCES:
        if not src.exists():
            print(f"missing source  {src}  -- skipping {fname}")
            continue
        dst = DST / fname
        sr, dur = _resample_and_clip(src, dst)
        meta.append({"file": fname, "duration_s": round(dur, 2), **info})
        print(f"wrote   {fname}  sr={sr}  dur={dur:.1f}s")
    (DST / "demos_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {DST / 'demos_meta.json'}  ({len(meta)} demos)")


if __name__ == "__main__":
    main()
