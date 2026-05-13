"""Regenerate the 4 production demos with Phase G defaults applied so the
strict G-11 tuplet-count audit can be done against the actually-shipped
state.

Source audio:
- bwv_854_piano: outputs/c5_vs_c5b_multi/bwv85.6_base.wav doesn't apply; the
  v3 bwv_854_piano render used /workspace/.cache/asap_renders/Bach__Fugue__bwv_854.wav
- maestro_chamber3_30s: outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1_30s.wav
- mtg_qbh_q1_humming: we don't have MTG-QBH audio on this host; skip and
  keep the prior render (humming branch is what Phase G impacts, and
  vocadito_1 already covers that case).
- vocadito_1_humming: /workspace/.cache/vocadito_orig/Audio/vocadito_1.wav
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe


DEMO_DIR = Path("outputs/demos")


def _run(audio: Path, input_kind: str, render_tpb: int, name: str,
          mode: str = "soft") -> None:
    cfg = PipelineConfig(
        input_kind=input_kind, mode=mode,
        pitch_model="pesto_crepevoicing" if input_kind == "humming" else "pesto",
        render_tpb=render_tpb,
        svg_path=str(DEMO_DIR / f"{name}.svg"),
        musicxml_path=str(DEMO_DIR / f"{name}.musicxml"),
    )
    transcribe(str(audio), cfg)
    print(f"wrote {DEMO_DIR / name}.{{svg,musicxml}}  (input_kind={input_kind}, render_tpb={render_tpb})")


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    bwv854 = Path("/workspace/.cache/asap_renders/Bach__Fugue__bwv_854.wav")
    if bwv854.exists():
        _run(bwv854, "piano", render_tpb=12, name="bwv_854_piano")
    maes = Path("outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1_30s.wav")
    if maes.exists():
        _run(maes, "piano", render_tpb=12, name="maestro_chamber3_30s")
    voc1 = Path("/workspace/.cache/vocadito_orig/Audio/vocadito_1.wav")
    if voc1.exists():
        _run(voc1, "humming", render_tpb=12, name="vocadito_1_humming")
    # mtg_qbh: keep prior render — no source audio on host.
    print("note: mtg_qbh_q1_humming not regenerated (source audio absent on host)")


if __name__ == "__main__":
    main()
