"""Phase E item 5: render JSB Chorales as (melody, arrangement) pairs.

For each Bach chorale in music21's corpus:
- soprano voice -> rendered as flute (program 73)  -> melody.wav
- all four voices -> rendered as organ (program 19) -> arrangement.wav
- both clipped to 15 s (the per-pair window we train MusicGen LoRA on)

Outputs land in /workspace/datasets/jsb_pairs/<id>/{melody.wav,arrangement.wav}.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from music21 import converter, instrument, note as m21note, stream
from music21.corpus.chorales import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("/workspace/datasets/jsb_pairs")
OUT.mkdir(parents=True, exist_ok=True)

# General MIDI program numbers
PROG_FLUTE = 73     # melody synth
PROG_ORGAN = 19     # arrangement synth (church organ)


def chorale_to_pretty_midi(score: stream.Score, only_soprano: bool,
                            program: int) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    parts = list(score.parts)
    sec_per_ql = 60.0 / 80.0  # JSB chorales typically ~80 bpm
    if only_soprano:
        parts = parts[:1]
    for p in parts:
        for el in p.recurse().notes:
            if isinstance(el, m21note.Note):
                start = float(el.offset) * sec_per_ql
                dur = float(el.quarterLength) * sec_per_ql
                if dur <= 0:
                    continue
                inst.notes.append(pretty_midi.Note(
                    velocity=80, pitch=int(el.pitch.midi),
                    start=start, end=start + dur,
                ))
            else:
                # Chord: split into per-pitch Note events
                start = float(el.offset) * sec_per_ql
                dur = float(el.quarterLength) * sec_per_ql
                if dur <= 0:
                    continue
                for p_obj in el.pitches:
                    inst.notes.append(pretty_midi.Note(
                        velocity=80, pitch=int(p_obj.midi),
                        start=start, end=start + dur,
                    ))
    pm.instruments.append(inst)
    return pm


def render_pair(score: stream.Score, out_dir: Path, sr: int = 32000,
                duration_s: float = 15.0) -> tuple[Path, Path] | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mel_p = out_dir / "melody.wav"
    arr_p = out_dir / "arrangement.wav"
    if mel_p.exists() and arr_p.exists():
        return mel_p, arr_p
    try:
        pm_mel = chorale_to_pretty_midi(score, only_soprano=True, program=PROG_FLUTE)
        pm_arr = chorale_to_pretty_midi(score, only_soprano=False, program=PROG_ORGAN)
        if not pm_mel.instruments or not pm_arr.instruments:
            return None
        mel = pm_mel.fluidsynth(fs=sr)
        arr = pm_arr.fluidsynth(fs=sr)
    except Exception as e:
        print(f"  fail: {e}"); return None
    n = int(duration_s * sr)
    mel = mel[:n]; arr = arr[:n]
    # pad if shorter
    if len(mel) < n:
        mel = np.concatenate([mel, np.zeros(n - len(mel))])
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    # normalise to ~-1 dBFS
    if (peak := np.max(np.abs(mel))) > 0:
        mel = mel / peak * 0.9
    if (peak := np.max(np.abs(arr))) > 0:
        arr = arr / peak * 0.9
    sf.write(str(mel_p), mel.astype(np.float32), sr)
    sf.write(str(arr_p), arr.astype(np.float32), sr)
    return mel_p, arr_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=400,
                    help="cap on number of chorales to render")
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--duration", type=float, default=15.0)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"Phase E item 5: JSB pair render -> {OUT}")
    print(f"sr={args.sr}, duration={args.duration}s, max={args.max}")
    print("=" * 60)

    it = Iterator()
    n_ok = 0; n_skip = 0
    for i, score in enumerate(it):
        if n_ok >= args.max:
            break
        bwv = str(score).split("bach/")[-1].rstrip(">").replace(".mxl", "")
        out_dir = OUT / bwv
        try:
            r = render_pair(score, out_dir, sr=args.sr, duration_s=args.duration)
        except Exception as e:
            print(f"[{i}] {bwv}: render exception {e}"); n_skip += 1; continue
        if r is None:
            n_skip += 1; continue
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"[{i}] {bwv}: {n_ok} rendered, {n_skip} skipped, "
                  f"wall={time.time()-t0:.1f}s")

    print(f"DONE: {n_ok} pairs in {time.time()-t0:.1f}s "
          f"(skipped {n_skip})")


if __name__ == "__main__":
    main()
