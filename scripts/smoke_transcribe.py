"""W-1 smoke transcription: confirm pipeline produces MIDI+MusicXML+SVG.

Reads app/demos/demo_1_vocadito_S1.wav (CC-BY public-domain humming clip,
ships with the repo) and writes outputs/smoke/* — produces no new model
artifacts, only proves the production path is wired.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe


def main() -> None:
    audio = Path("app/demos/demo_1_vocadito_S1.wav")
    if not audio.exists():
        raise SystemExit(f"missing audio at {audio}")
    out_root = Path("outputs/smoke")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(
        input_kind="humming",
        mode="soft",
        pitch_model="pesto_crepevoicing",
        svg_path=str(out_root / "smoke.svg"),
        musicxml_path=str(out_root / "smoke.musicxml"),
    )
    r = transcribe(str(audio), cfg)
    have_svg = Path(cfg.svg_path).exists()
    have_mxl = Path(cfg.musicxml_path).exists()
    print(f"notes={r.n_notes}  bpm={r.bpm:.2f}")
    print(f"svg_exists={have_svg}  mxl_exists={have_mxl}")
    print(f"svg={cfg.svg_path}")
    print(f"mxl={cfg.musicxml_path}")


if __name__ == "__main__":
    main()
