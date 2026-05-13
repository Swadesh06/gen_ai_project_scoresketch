"""Regenerate demo SVGs before/after the Phase G post-processing defaults.

`*_before.svg` = pipeline with G-4/5/6/11 forcibly OFF (mirrors pre-Phase-G
production state).
`*_after.svg` = pipeline with the Phase G defaults intact (G-4/5/6 = auto on
humming, G-11 render_tpb_auto = auto).

Only the humming demo (vocadito_1) is regenerated because the piano /
chamber / mtg_qbh demos use the instrument branch, which Phase G post-
processing doesn't touch.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe

OUT = Path("outputs/demos")
VOC1 = Path("/workspace/.cache/vocadito_orig/Audio/vocadito_1.wav")


def _run(audio: Path, cfg: PipelineConfig, svg_path: Path, mxl_path: Path) -> None:
    cfg.svg_path = str(svg_path)
    cfg.musicxml_path = str(mxl_path)
    transcribe(str(audio), cfg)
    print(f"wrote {svg_path}")


def main() -> None:
    # First save the existing vocadito_1_humming.svg as the "_before.svg" to
    # preserve the visible record of the pre-Phase-G state, then regenerate
    # both states freshly so the comparison is from the same audio.
    existing_svg = OUT / "vocadito_1_humming.svg"
    existing_mxl = OUT / "vocadito_1_humming.musicxml"
    if existing_svg.exists() and not (OUT / "vocadito_1_humming_pre_phase_g.svg").exists():
        shutil.copy(existing_svg, OUT / "vocadito_1_humming_pre_phase_g.svg")
        shutil.copy(existing_mxl, OUT / "vocadito_1_humming_pre_phase_g.musicxml")

    # before: force all Phase G post-processing off; mirror pre-Phase-G defaults
    # (formant_offset stayed off in production; we keep that).
    cfg_before = PipelineConfig(
        input_kind="humming", mode="soft", pitch_model="pesto_crepevoicing",
        same_pitch_merge="off", median_smooth_g5="off", silent_trim_g6="off",
        render_tpb_auto="off",
    )
    _run(VOC1, cfg_before,
         OUT / "vocadito_1_humming_before.svg",
         OUT / "vocadito_1_humming_before.musicxml")

    # after: Phase G production defaults (auto on all G-4/5/6/11).
    cfg_after = PipelineConfig(
        input_kind="humming", mode="soft", pitch_model="pesto_crepevoicing",
    )
    _run(VOC1, cfg_after,
         OUT / "vocadito_1_humming_after.svg",
         OUT / "vocadito_1_humming_after.musicxml")


if __name__ == "__main__":
    main()
