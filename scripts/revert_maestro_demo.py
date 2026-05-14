"""W-1: Revert outputs/demos/maestro_chamber3_30s.{svg,musicxml} to the
pre-Phase-G clean state.

Phase G's G-11 render_tpb auto-detect does not fire on the chamber piece, so
the Phase G demo regen produced an SVG with tempo 154 + 4x 24-lets + 1x 48-let.
This script forces render_tpb=8 (the value Phase E v3 item 8 had explicitly
used) so no tuplet denser than a quadruplet can appear.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe


def main() -> None:
    audio = Path(
        "outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1_30s.wav"
    )
    if not audio.exists():
        raise SystemExit(f"missing audio at {audio}")
    out_root = Path("outputs/demos")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        input_kind="instrument",
        mode="soft",
        tatums_per_beat=8,
        render_tpb=8,
        svg_path=str(out_root / "maestro_chamber3_30s.svg"),
        musicxml_path=str(out_root / "maestro_chamber3_30s.musicxml"),
    )
    r = transcribe(str(audio), cfg)
    mxl = Path(cfg.musicxml_path).read_text()
    n_24 = mxl.count("<actual-notes>24</actual-notes>")
    n_48 = mxl.count("<actual-notes>48</actual-notes>")
    print(f"notes={r.n_notes}  bpm={r.bpm:.2f}")
    print(f"<actual-notes>24 occurrences: {n_24}")
    print(f"<actual-notes>48 occurrences: {n_48}")


if __name__ == "__main__":
    main()
