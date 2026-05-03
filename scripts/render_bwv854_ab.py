"""Render Bach BWV 854 SVG with old vs new pipeline behavior for visual A/B (item 1.5)."""
from __future__ import annotations
import subprocess, json
from pathlib import Path
import numpy as np
import pretty_midi
from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import quantize_with_voice_tracking
from humscribe.score import build_stream, render_svg

ASAP = Path("~/datasets/asap").expanduser()
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
RENDER = Path("/workspace/.cache/asap_renders")
OUT = Path("outputs/item1_ab")
OUT.mkdir(parents=True, exist_ok=True)


def render(midi: Path, wav: Path) -> None:
    if wav.exists() and wav.stat().st_size > 0: return
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)], check=True, capture_output=True)


def main() -> None:
    piece = ASAP / "Bach/Fugue/bwv_854"
    mid = piece / "midi_score.mid"
    wav = RENDER / "Bach__Fugue__bwv_854.wav"
    render(mid, wav)
    audio, sr = load_audio(str(wav), target_sr=22050)
    notes = transcribe_piano(str(wav))
    beats, _, bpm = track_beats_beat_this(str(wav))
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    print(f"BWV 854: {len(notes)} notes, bpm={bpm:.2f}")

    # Before: prune_unreadable=False, render_tpb=24, estimate_key=False
    q_on_b, q_off_b = quantize_with_voice_tracking(notes, beats, tatums_per_beat=24)
    # Manually call viterbi w/ no prune for the "before" baseline visual
    from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations
    cfg = VoiceTrackConfig(pitch_jump=adaptive_pitch_jump(notes), time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on_old, q_off_old = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=24, prune_unreadable=False)
    s_old = build_stream(notes, bpm=bpm, time_sig="4/4",
                         tatum_onsets=q_on_old, tatum_offsets=q_off_old,
                         tatums_per_beat=24, render_tpb=24, estimate_key=False)
    svg_old = render_svg(s_old, notes, bpm)
    (OUT / "bwv_854_BEFORE.svg").write_text(svg_old)

    # After: defaults (prune=True, render_tpb=12, estimate_key=True)
    q_on_new, q_off_new = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=24, prune_unreadable=True)
    s_new = build_stream(notes, bpm=bpm, time_sig="4/4",
                         tatum_onsets=q_on_new, tatum_offsets=q_off_new,
                         tatums_per_beat=24, render_tpb=12, estimate_key=True)
    svg_new = render_svg(s_new, notes, bpm)
    (OUT / "bwv_854_AFTER.svg").write_text(svg_new)

    print(f"wrote {OUT}/bwv_854_BEFORE.svg  ({len(svg_old)} bytes)")
    print(f"wrote {OUT}/bwv_854_AFTER.svg   ({len(svg_new)} bytes)")
    # Lightweight quantitative diff: count how many tuplet markers in each
    def count_tuplets(svg: str) -> dict:
        out = {}
        for d in (3, 5, 6, 7, 12, 24, 48):
            # Verovio renders tuplet brackets with class containing the number; rough heuristic
            out[d] = svg.lower().count(f">{d}<")
        return out
    print("BEFORE tuplet-text occurrences:", count_tuplets(svg_old))
    print("AFTER  tuplet-text occurrences:", count_tuplets(svg_new))


if __name__ == "__main__":
    main()
