"""Render real-notation SVGs for one example each from Vocadito, MTG-QBH,
ASAP Bach, and MAESTRO (demo set for quick visual inspection)."""
from __future__ import annotations
import os
import sys
from pathlib import Path

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe


CLIPS = [
    ("vocadito_1_humming",  "/home/swadesh/datasets/vocadito/Audio/vocadito_1.wav",       "humming"),
    ("mtg_qbh_q1_humming",  "/home/swadesh/datasets/mtg_qbh/audio/q1.wav",                "humming"),
    ("bwv_854_piano",       "/home/swadesh/datasets/asap/Bach/Fugue/bwv_854/midi_score.wav", "piano"),
]


def maybe_add_maestro() -> None:
    out = Path("outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--2_30s.wav")
    if out.exists():
        CLIPS.append(("maestro_chamber3_30s", str(out), "piano"))


def main() -> None:
    maybe_add_maestro()
    out_root = Path("outputs/demos")
    out_root.mkdir(parents=True, exist_ok=True)
    for name, audio, kind in CLIPS:
        if not Path(audio).exists():
            print(f"skip {name}: no audio at {audio}")
            continue
        cfg = PipelineConfig(
            input_kind=kind, mode="soft" if kind == "humming" else "medium",
            svg_path=str(out_root / f"{name}.svg"),
            musicxml_path=str(out_root / f"{name}.musicxml"),
        )
        try:
            r = transcribe(audio, cfg)
        except Exception as e:
            print(f"{name}: failed -- {e}")
            continue
        size_svg = os.path.getsize(out_root / f"{name}.svg")
        print(f"{name:30s}  notes={r.n_notes:4d}  bpm={r.bpm:6.1f}  svg={size_svg/1024:.1f}KB")


if __name__ == "__main__":
    main()
