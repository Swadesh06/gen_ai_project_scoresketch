"""Phase E item 6: pre-cache features for the MV2H-driven sweep.

The sweep needs to evaluate ~100 configs of (DP params, voicing thresholds,
TPB, target_bpm correction) on 5 ASAP + 10 Vocadito pieces, with MV2H as
the optimization target. Re-running PESTO/CREPE/ByteDance/YMT3 for each
sweep run wastes time. Instead, we cache:

- PESTO and CREPE outputs per humming clip (for the voicing pipeline)
- ByteDance outputs per ASAP piece (for the piano pipeline)
- beat_this outputs per piece (for both)
- GT MIDIs converted to MV2H text once

The sweep agents then read these cached features, run only the DP /
rendering / metric path (CPU), and evaluate MV2H. 6 parallel CPU agents
each completes a run in ~5 s, for a sweep of 100 runs in ~10 min total.

Outputs:
- /workspace/.cache/sweep_e6_features/<piece>.npz
- /workspace/.cache/sweep_e6_features/<piece>_gt.txt
"""
from __future__ import annotations
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent

CACHE = Path("/workspace/.cache/sweep_e6_features")
CACHE.mkdir(parents=True, exist_ok=True)

# 5 ASAP + 10 Vocadito eval set, per task_description_v3.md item 6.
ASAP_KEYS = [
    "Bach__Fugue__bwv_854",      # clean fugue, high-quality piece
    "Beethoven__Piano_Sonatas__21-1",  # Romantic, B76 helps
    "Chopin__Berceuse_op_57",    # Chopin-style melody+accomp
    "Liszt__Sonata",             # structurally hard
    "Schumann__Toccata",         # dense Romantic
]

VOC_IDS = list(range(1, 11))  # vocadito_1..vocadito_10

ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
ASAP_PIECES = {
    "Bach__Fugue__bwv_854": "Bach/Fugue/bwv_854",
    "Bach__Fugue__bwv_846": "Bach/Fugue/bwv_846",
    "Bach__Fugue__bwv_848": "Bach/Fugue/bwv_848",
    "Bach__Fugue__bwv_856": "Bach/Fugue/bwv_856",
    "Bach__Fugue__bwv_857": "Bach/Fugue/bwv_857",
    "Beethoven__Piano_Sonatas__21-1": "Beethoven/Piano_Sonatas/21-1",
    "Chopin__Berceuse_op_57": "Chopin/Berceuse_op_57",
    "Liszt__Sonata": "Liszt/Sonata",
    "Schumann__Toccata": "Schumann/Toccata",
}
ASAP_YMT3_DIR = Path("/workspace/.cache/asap_yourmt3plus")
ASAP_RENDER_DIR = Path("/workspace/.cache/asap_renders")
VOC_AUDIO_DIR = Path("/home/swadesh/datasets/vocadito/Audio")
VOC_ANN_DIR = Path("/home/swadesh/datasets/vocadito/Annotations/Notes")


def _ymt3_to_pipeline_notes(notes_dicts: list[dict]) -> list[NoteEvent]:
    out = []
    for n in notes_dicts:
        midi = int(n["midi"])
        if midi < 1 or midi > 127:
            continue
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        out.append(NoteEvent(onset_s=float(n["on"]), offset_s=float(n["off"]),
                              pitch_midi=midi, pitch_hz=hz,
                              velocity=int(n.get("vel", 80)), confidence=1.0))
    out.sort(key=lambda e: e.onset_s)
    return out


def cache_asap(eval_seconds: float | None) -> None:
    """Cache ASAP features: YMT3 notes + beat_this beats + GT MV2H text."""
    import pretty_midi
    for key, piece_dir in ASAP_PIECES.items():
        out_npz = CACHE / f"asap_{key}.npz"
        out_gt = CACHE / f"asap_{key}_gt.txt"
        if out_npz.exists() and out_gt.exists():
            print(f"have {key}")
            continue
        t0 = time.time()
        pkl_path = ASAP_YMT3_DIR / f"{key}.pkl"
        if not pkl_path.exists():
            print(f"skip {key}: no YMT3 cache"); continue
        with open(pkl_path, "rb") as f:
            cache = pickle.load(f)
        notes = _ymt3_to_pipeline_notes(cache["notes"])
        if eval_seconds is not None:
            notes = [n for n in notes if n.onset_s < eval_seconds]
        # beats: re-run beat_this on the rendered audio
        audio_path = ASAP_RENDER_DIR / f"{key}.wav"
        if not audio_path.exists():
            print(f"skip {key}: no rendered audio"); continue
        beats, downbeats, bpm = track_beats_beat_this(
            str(audio_path), target_bpm=110.0,
        )
        np.savez(str(out_npz),
                 notes_on=np.array([n.onset_s for n in notes], dtype=np.float64),
                 notes_off=np.array([n.offset_s for n in notes], dtype=np.float64),
                 notes_midi=np.array([n.midi() for n in notes], dtype=np.int32),
                 beats=beats, downbeats=downbeats,
                 bpm=np.array([bpm], dtype=np.float64))

        # GT MV2H text from midi_score.mid
        gt_path = ASAP_REPO / piece_dir / "midi_score.mid"
        if gt_path.exists():
            pm = pretty_midi.PrettyMIDI(str(gt_path))
            gt_bpm = float(pm.estimate_tempo()) if pm.instruments else 120.0
            ts_changes = pm.time_signature_changes
            ts = f"{ts_changes[0].numerator}/{ts_changes[0].denominator}" if ts_changes else "4/4"
            gt_notes: list[NoteEvent] = []; voices: list[int] = []
            for ti, inst in enumerate(pm.instruments):
                for n in inst.notes:
                    if eval_seconds is not None and n.start >= eval_seconds:
                        continue
                    gt_notes.append(NoteEvent(onset_s=float(n.start), offset_s=float(n.end),
                                              pitch_midi=int(n.pitch), velocity=int(n.velocity)))
                    voices.append(ti)
            txt = notes_to_mv2h_format(gt_notes, bpm=gt_bpm, time_sig=ts, voices=voices)
            out_gt.write_text(txt)
        else:
            print(f"warning: no GT MIDI for {key}")
        print(f"cached {key} in {time.time()-t0:.1f}s "
              f"(notes={len(notes)} bpm_pred={bpm:.1f})")


def cache_vocadito(eval_seconds: float | None) -> None:
    """Cache Vocadito features: PESTO+CREPE traces + GT MV2H text per annotator."""
    from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
    for vid in VOC_IDS:
        out_npz = CACHE / f"voc_{vid}.npz"
        if out_npz.exists():
            print(f"have voc_{vid}")
            continue
        audio = VOC_AUDIO_DIR / f"vocadito_{vid}.wav"
        if not audio.exists():
            print(f"skip voc_{vid}: no audio"); continue
        t0 = time.time()
        y, sr = load_audio(str(audio), target_sr=22050)
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        beats, downbeats, bpm = track_beats_beat_this(str(audio), target_bpm=110.0)
        np.savez(str(out_npz),
                 t=t.astype(np.float32), hz=hz.astype(np.float32),
                 vc=vc.astype(np.float32),
                 beats=beats, downbeats=downbeats,
                 bpm=np.array([bpm], dtype=np.float64))
        # GT MV2H text per annotator
        for ann in ("A1", "A2"):
            csv = VOC_ANN_DIR / f"vocadito_{vid}_notes{ann}.csv"
            if not csv.exists():
                continue
            notes = []
            for line in csv.read_text().splitlines():
                if not line.strip(): continue
                a, b, c = line.split(",")
                on = float(a); freq = float(b); dur = float(c)
                if freq <= 0: continue
                midi = int(round(69 + 12 * np.log2(freq / 440.0)))
                if eval_seconds is not None and on >= eval_seconds: continue
                notes.append(NoteEvent(onset_s=on, offset_s=on + max(dur, 1e-3),
                                       pitch_midi=midi, pitch_hz=freq, velocity=80))
            iois = np.diff([n.onset_s for n in notes]) if len(notes) >= 2 else np.array([0.5])
            est_bpm = 60.0 / max(float(np.median(iois)), 0.1)
            txt = notes_to_mv2h_format(notes, bpm=est_bpm, time_sig="4/4",
                                       voices=[0]*len(notes))
            (CACHE / f"voc_{vid}_{ann}_gt.txt").write_text(txt)
        print(f"cached voc_{vid} in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--asap-only", action="store_true")
    ap.add_argument("--voc-only", action="store_true")
    args = ap.parse_args()
    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    print("=" * 60)
    print(f"Phase E item 6 cache prep — eval_seconds={eval_sec}")
    print(f"output dir: {CACHE}")
    print("=" * 60)
    if not args.voc_only:
        cache_asap(eval_sec)
    if not args.asap_only:
        cache_vocadito(eval_sec)
    print("done")


if __name__ == "__main__":
    main()
