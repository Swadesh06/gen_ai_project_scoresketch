"""Re-render MAESTRO chamber demo with render_tpb=8 to eliminate 24-lets.

v3 item 8 strict pass criterion: zero 24-lets or 48-lets. Current output
at render_tpb=12 still has 2x 24-lets from music21 makeNotation deciding
to subdivide on ties. Drop render_tpb to 8 to ensure the smallest tuplet
is a quadruplet.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe


def main():
    audio = "outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--2_30s.wav"
    if not Path(audio).exists():
        print(f"missing audio at {audio}")
        return
    out_root = Path("outputs/demos")
    out_root.mkdir(parents=True, exist_ok=True)
    # render_tpb=8 -> no sextuplets at render time -> no 24-let possible
    cfg = PipelineConfig(
        input_kind="piano", mode="medium",
        tatums_per_beat=8, render_tpb=8,
        svg_path=str(out_root / "maestro_chamber3_30s.svg"),
        musicxml_path=str(out_root / "maestro_chamber3_30s.musicxml"),
    )
    r = transcribe(audio, cfg)
    svg = Path(cfg.svg_path).read_text()
    n_24 = svg.count('class="tuplet"') if 'class="tuplet"' in svg else 0
    n_dense = sum(svg.count(f'num="{n}"') for n in (24, 48))
    print(f"notes={r.n_notes}  bpm={r.bpm:.2f}  svg_size={len(svg)/1024:.1f}KB")
    print(f"24-let/48-let occurrences in SVG: {n_dense}")


if __name__ == "__main__":
    main()
