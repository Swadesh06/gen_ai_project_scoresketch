"""Render Bach BWV 854 SVG with the new YMT3+ default for the item-6 demo figure.

Compares the rendered SVG produced by the *full new pipeline*:
  ByteDance + render_tpb=24 + no key  (legacy)
vs
  YourMT3+ + render_tpb=12 + KS key   (current default)
on the same Bach BWV 854 audio. Outputs both SVGs and a side-by-side HTML.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.instrument.yourmt3plus import transcribe_yourmt3plus
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import (
    VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations,
)
from humscribe.score import build_stream, render_svg

ASAP = Path("~/datasets/asap").expanduser()
RENDER = Path("/workspace/.cache/asap_renders")
OUT = Path("outputs/item6_bwv854_demo")
OUT.mkdir(parents=True, exist_ok=True)


def quantize(notes, beats, prune):
    cfg = VoiceTrackConfig(pitch_jump=adaptive_pitch_jump(notes), time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    return viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=24,
                                    prune_unreadable=prune)


def main() -> None:
    piece = ASAP / "Bach/Fugue/bwv_854"
    src_wav = piece / "midi_score.wav"
    cached_wav = RENDER / "Bach__Fugue__bwv_854.wav"
    wav = src_wav if src_wav.exists() else cached_wav
    print(f"audio: {wav}")
    beats, _, bpm = track_beats_beat_this(str(wav))
    print(f"bpm={bpm:.2f}, n_beats={len(beats)}")

    notes_bd = transcribe_piano(str(wav))
    print(f"BD notes: {len(notes_bd)}")
    q_on_bd, q_off_bd = quantize(notes_bd, beats, prune=False)
    s_bd = build_stream(notes_bd, bpm=bpm, time_sig="4/4",
                         tatum_onsets=q_on_bd, tatum_offsets=q_off_bd,
                         tatums_per_beat=24, render_tpb=24, estimate_key=False)
    svg_bd = render_svg(s_bd, notes_bd, bpm)
    (OUT / "bwv_854_BD_legacy.svg").write_text(svg_bd)

    notes_ymt = transcribe_yourmt3plus(str(wav))
    print(f"YMT3 notes: {len(notes_ymt)}")
    q_on_ymt, q_off_ymt = quantize(notes_ymt, beats, prune=True)
    s_ymt = build_stream(notes_ymt, bpm=bpm, time_sig="4/4",
                          tatum_onsets=q_on_ymt, tatum_offsets=q_off_ymt,
                          tatums_per_beat=24, render_tpb=12, estimate_key=True)
    svg_ymt = render_svg(s_ymt, notes_ymt, bpm)
    (OUT / "bwv_854_YMT3_current.svg").write_text(svg_ymt)

    template = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:-apple-system,sans-serif;margin:24px}"
        ".row{display:flex;gap:16px;margin-bottom:24px}"
        ".col{flex:1;border:1px solid #ddd;padding:12px;overflow:auto}"
        "h2{margin-top:0;font-size:14px}svg{max-width:100%}</style></head><body>"
        "<h1>Bach BWV 854 — pipeline before vs after (B+2)</h1>"
        "<div class='row'>"
        "<div class='col'><h2>BEFORE: ByteDance + render_tpb=24 + no key</h2>__BD__</div>"
        "<div class='col'><h2>AFTER: YourMT3+ + render_tpb=12 + KS key</h2>__YMT__</div>"
        "</div></body></html>"
    )
    html = template.replace("__BD__", svg_bd).replace("__YMT__", svg_ymt)
    (OUT / "compare.html").write_text(html)

    def count_text(svg, needles):
        s = svg.lower()
        return {n: s.count(n) for n in needles}
    needles = [">24<", ">48<", ">7<", ">5<", ">3<"]
    print("BEFORE:", count_text(svg_bd, needles))
    print("AFTER :", count_text(svg_ymt, needles))


if __name__ == "__main__":
    main()
