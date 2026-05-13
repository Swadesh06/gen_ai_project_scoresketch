"""Phase G G-1 + G-2 + (later) G-3..G-6 MV2H eval.

Unified eval driver that adds:
- voice IDs computed via B76 (piano/instrument) or greedy fallback
- tatum positions interpolated from real beat positions (beat_this on pred,
  pretty_midi.get_beats() on GT)

Runs the same 9-piece ASAP set, the 5-clip MAESTRO chamber set, and the
40-clip Vocadito A1 set. Emits per-piece per-axis MV2H to
`reports/_metric_mv2h_phase_g_{dataset}.json` and a headline JSON at
`reports/_item-g{1,2}.json`.

Sources:
- ASAP pred: cached YourMT3+ (.pkl in /workspace/.cache/asap_yourmt3plus)
  + cached beats (run-once via beat_this on /workspace/.cache/asap_renders)
- MAESTRO pred: pipeline.transcribe on outputs/maestro_clips/*.wav
- Vocadito pred: pipeline.transcribe on /home/swadesh/datasets/vocadito/Audio
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.eval.voice_emission import voice_ids_for_emission
from humscribe.notes import NoteEvent

CACHE_YMT3 = Path("/workspace/.cache/asap_yourmt3plus")
CACHE_RENDERS = Path("/workspace/.cache/asap_renders")
CACHE_BEATS = Path("/workspace/.cache/asap_beats")
ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
MAESTRO_CLIPS = Path("outputs/maestro_clips")
VOC_DIR = Path("/home/swadesh/datasets/vocadito")

ASAP_PIECES = [
    ("Bach/Fugue/bwv_846", "Bach__Fugue__bwv_846"),
    ("Bach/Fugue/bwv_848", "Bach__Fugue__bwv_848"),
    ("Bach/Fugue/bwv_854", "Bach__Fugue__bwv_854"),
    ("Bach/Fugue/bwv_856", "Bach__Fugue__bwv_856"),
    ("Bach/Fugue/bwv_857", "Bach__Fugue__bwv_857"),
    ("Beethoven/Piano_Sonatas/21-1", "Beethoven__Piano_Sonatas__21-1"),
    ("Schumann/Toccata", "Schumann__Toccata"),
    ("Chopin/Berceuse_op_57", "Chopin__Berceuse_op_57"),
    ("Liszt/Sonata", "Liszt__Sonata"),
]


def _ymt3_notes(piece_key: str, eval_seconds: float | None) -> tuple[list[NoteEvent], float]:
    p = CACHE_YMT3 / f"{piece_key}.pkl"
    if not p.exists():
        return [], 120.0
    with open(p, "rb") as f:
        cache = pickle.load(f)
    bpm = float(cache.get("bpm", 120.0))
    notes: list[NoteEvent] = []
    for n in cache["notes"]:
        midi = int(n["midi"])
        if midi < 1 or midi > 127:
            continue
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        notes.append(NoteEvent(
            onset_s=float(n["on"]), offset_s=float(n["off"]),
            pitch_midi=midi, pitch_hz=hz,
            velocity=int(n.get("vel", 80)), confidence=float(n.get("conf", 1.0)),
        ))
    notes.sort(key=lambda e: e.onset_s)
    if eval_seconds is not None:
        notes = [n for n in notes if n.onset_s < eval_seconds]
    return notes, bpm


def _cached_beats_or_compute(piece_key: str) -> tuple[np.ndarray, np.ndarray, float]:
    CACHE_BEATS.mkdir(parents=True, exist_ok=True)
    p = CACHE_BEATS / f"{piece_key}.npz"
    if p.exists():
        d = np.load(str(p))
        return d["beats"], d["downbeats"], float(d["bpm"])
    wav = CACHE_RENDERS / f"{piece_key}.wav"
    if not wav.exists():
        return np.zeros(0), np.zeros(0), 120.0
    from humscribe.beat.beat_this_track import track_beats_beat_this
    beats, downbeats, bpm = track_beats_beat_this(str(wav), target_bpm=110.0)
    # Apply F-1 octave-sanity correction using the cached YMT3 notes (matches
    # the production pipeline path so the beat positions we emit are aligned
    # with what production produces).
    notes, _ = _ymt3_notes(piece_key, eval_seconds=None)
    if len(notes) > 0:
        try:
            from humscribe.beat.octave_sanity import (
                detect_octave_misalignment, apply_octave_correction,
            )
            diag = detect_octave_misalignment(beats, notes)
            if diag["recommend"] != "keep":
                beats, downbeats = apply_octave_correction(beats, downbeats, diag["recommend"])
                if len(beats) >= 2:
                    ibis = np.diff(beats)
                    ibis = ibis[(ibis > 0.01) & (ibis < 5.0)]
                    if len(ibis) > 0:
                        bpm = 60.0 / float(np.median(ibis))
        except Exception:
            pass
    np.savez(str(p), beats=beats, downbeats=downbeats, bpm=np.float64(bpm))
    return beats, downbeats, bpm


def _gt_text(mid_path: Path, eval_seconds: float | None, *,
             use_beats: bool, use_voices: bool, tatums_per_beat: int = 4) -> tuple[str, dict] | None:
    """GT text always preserves the score's multi-voice track structure and
    its native beat positions. The `use_voices`/`use_beats` flags control
    only the PRED side; GT keeps full signal so the prior baseline rows
    (which gave GT voices = track ids) remain apples-to-apples."""
    if not mid_path.exists():
        return None
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    bpm = float(pm.estimate_tempo()) if pm.instruments else 120.0
    ts_changes = pm.time_signature_changes
    ts = f"{ts_changes[0].numerator}/{ts_changes[0].denominator}" if ts_changes else "4/4"
    notes: list[NoteEvent] = []
    voices: list[int] = []
    for ti, inst in enumerate(pm.instruments):
        for n in inst.notes:
            if eval_seconds is not None and n.start >= eval_seconds:
                continue
            notes.append(NoteEvent(onset_s=float(n.start), offset_s=float(n.end),
                                   pitch_midi=int(n.pitch), velocity=int(n.velocity)))
            voices.append(ti)
    beats = None
    if use_beats:
        # Use the GT's native beat positions for the tatum grid. This keeps
        # both sides on a comparable real-tempo axis (instead of the
        # uniform-from-bpm fallback that crushes meter on rubato pieces).
        try:
            beats = list(pm.get_beats())
        except Exception:
            beats = None
        if beats is not None and eval_seconds is not None:
            beats = [b for b in beats if b < eval_seconds + 5.0]
    txt = notes_to_mv2h_format(
        notes, bpm=bpm, time_sig=ts, voices=voices, tatums_per_beat=tatums_per_beat,
        beats=beats,
    )
    return txt, {"bpm": bpm, "time_sig": ts, "n_notes_gt": len(notes), "n_beats_gt": int(len(beats)) if beats is not None else 0}


def _pred_text_asap(piece_key: str, eval_seconds: float | None, *,
                     use_beats: bool, use_voices: bool, tatums_per_beat: int = 4) -> tuple[str, dict]:
    notes, bpm = _ymt3_notes(piece_key, eval_seconds)
    beats_arr = None
    if use_beats:
        beats, _db, b_bpm = _cached_beats_or_compute(piece_key)
        beats_arr = beats[beats < (eval_seconds + 5.0 if eval_seconds else 1e9)]
        if b_bpm > 0:
            bpm = b_bpm
    voices = (voice_ids_for_emission(notes, input_kind="piano") if use_voices
              else [0] * len(notes))
    txt = notes_to_mv2h_format(
        notes, bpm=bpm, time_sig="4/4", voices=voices,
        tatums_per_beat=tatums_per_beat, beats=beats_arr,
    )
    return txt, {"bpm": bpm, "n_notes_pred": len(notes),
                  "n_voices_pred": int(max(voices) + 1) if voices else 0}


def _run_asap(args, *, use_beats: bool, use_voices: bool) -> dict:
    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    rows = []
    skipped = []
    for piece_dir, piece_key in ASAP_PIECES:
        if args.piece and piece_key != args.piece:
            continue
        gt = _gt_text(ASAP_REPO / piece_dir / "midi_score.mid", eval_sec,
                      use_beats=use_beats, use_voices=use_voices,
                      tatums_per_beat=args.tatums_per_beat)
        if gt is None:
            skipped.append({"piece": piece_key, "reason": "no_gt"}); continue
        gt_text, gt_info = gt
        pred_text, pred_info = _pred_text_asap(
            piece_key, eval_sec,
            use_beats=use_beats, use_voices=use_voices,
            tatums_per_beat=args.tatums_per_beat,
        )
        try:
            res = compute_mv2h(pred_text, gt_text, align="non_aligned")
        except Exception as e:
            skipped.append({"piece": piece_key, "reason": f"mv2h_exception: {e}"}); continue
        row = {"piece": piece_key, **res.as_dict(),
               "n_notes_gt": gt_info["n_notes_gt"],
               "n_notes_pred": pred_info["n_notes_pred"],
               "n_voices_pred": pred_info.get("n_voices_pred"),
               "n_beats_gt": gt_info.get("n_beats_gt", 0)}
        rows.append(row)
        print(f"asap {piece_key:42s} mv2h={res.mv2h:.4f}  "
              f"mp={res.multi_pitch:.3f} v={res.voice:.3f} "
              f"m={res.meter:.3f} val={res.value:.3f} h={res.harmony:.3f} "
              f"(gt={gt_info['n_notes_gt']} pred={pred_info['n_notes_pred']} vid={pred_info.get('n_voices_pred')})")
    mean = {k: float(np.nanmean([r[k] for r in rows]))
            for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony")} if rows else {}
    return {"rows": rows, "mean": mean, "skipped": skipped,
            "use_beats": use_beats, "use_voices": use_voices,
            "alignment": "non_aligned", "eval_seconds": eval_sec,
            "tatums_per_beat": args.tatums_per_beat}


def _run_maestro(args, *, use_beats: bool, use_voices: bool, phase_g_post: bool = False) -> dict:
    from humscribe.pipeline import transcribe
    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    rows = []
    for wav in sorted(MAESTRO_CLIPS.glob("*.wav")):
        mid = wav.with_suffix(".mid")
        if not mid.exists():
            continue
        gt = _gt_text(mid, eval_sec, use_beats=use_beats, use_voices=use_voices,
                      tatums_per_beat=args.tatums_per_beat)
        if gt is None:
            continue
        gt_text, gt_info = gt
        cfg = PipelineConfig(input_kind="piano", mode="medium", render_svg=False,
                             per_voice_dp="off", transcriber="bytedance_piano",
                             same_pitch_merge="auto" if phase_g_post else "off",
                             median_smooth_g5="auto" if phase_g_post else "off",
                             silent_trim_g6="auto" if phase_g_post else "off")
        r = transcribe(str(wav), cfg)
        notes = list(r.notes)
        if eval_sec is not None:
            notes = [n for n in notes if n.onset_s < eval_sec]
        voices = (voice_ids_for_emission(notes, input_kind="piano") if use_voices
                  else [0] * len(notes))
        beats_arr = None
        if use_beats and r.beats is not None and len(r.beats) >= 2:
            beats_arr = [b for b in r.beats if b < (eval_sec + 5.0 if eval_sec else 1e9)]
        pred_text = notes_to_mv2h_format(
            notes, bpm=float(r.bpm) if r.bpm else 120.0,
            time_sig="4/4", voices=voices,
            tatums_per_beat=args.tatums_per_beat, beats=beats_arr,
        )
        try:
            res = compute_mv2h(pred_text, gt_text, align="aligned")
        except Exception as e:
            print(f"skip {wav.name}: {e}"); continue
        row = {"clip": wav.name, **res.as_dict(),
               "n_notes_gt": gt_info["n_notes_gt"], "n_notes_pred": len(notes),
               "bpm_pred": float(r.bpm),
               "n_voices_pred": int(max(voices) + 1) if voices else 0}
        rows.append(row)
        print(f"maes {wav.name:48s} mv2h={res.mv2h:.4f} v={res.voice:.3f} m={res.meter:.3f} mp={res.multi_pitch:.3f} val={res.value:.3f}")
    mean = {k: float(np.nanmean([r[k] for r in rows]))
            for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony")} if rows else {}
    return {"rows": rows, "mean": mean,
            "use_beats": use_beats, "use_voices": use_voices,
            "alignment": "aligned"}


def _voc_load_gt(clip_name: str, annotator: str = "A1") -> tuple[list[NoteEvent], float] | None:
    """Vocadito GT: per-clip CSV under Annotations/Notes/<clip>_notes<A1|A2>.csv."""
    csv = VOC_DIR / "Annotations" / "Notes" / f"{clip_name}_notes{annotator}.csv"
    if not csv.exists():
        return None
    notes: list[NoteEvent] = []
    for line in csv.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            on = float(parts[0]); hz = float(parts[1]); dur = float(parts[2])
        except ValueError:
            continue
        if hz <= 0:
            continue
        midi = int(round(69 + 12 * np.log2(hz / 440.0)))
        notes.append(NoteEvent(onset_s=on, offset_s=on + max(dur, 1e-3),
                                pitch_midi=midi, pitch_hz=hz, velocity=80))
    bpm = 120.0  # vocadito unannotated tempo; bpm only affects fallback grid
    return notes, bpm


def _run_vocadito(args, *, use_beats: bool, use_voices: bool,
                   annotator: str = "A1", phase_g_post: bool = False) -> dict:
    from humscribe.pipeline import transcribe
    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    audio_dir = VOC_DIR / "Audio"
    rows = []
    wavs = sorted(audio_dir.glob("vocadito_*.wav"))
    if getattr(args, "limit", 0) > 0:
        wavs = wavs[: args.limit]
    for wav in wavs:
        clip = wav.stem
        gt = _voc_load_gt(clip, annotator=annotator)
        if gt is None:
            continue
        gt_notes, gt_bpm = gt
        if eval_sec is not None:
            gt_notes = [n for n in gt_notes if n.onset_s < eval_sec]
        # GT side: humming, voices all 0; no beats (vocadito GT has no beat track).
        gt_voices = [0] * len(gt_notes)
        gt_text = notes_to_mv2h_format(
            gt_notes, bpm=gt_bpm, time_sig="4/4", voices=gt_voices,
            tatums_per_beat=args.tatums_per_beat,
        )
        cfg = PipelineConfig(input_kind="humming", mode="soft",
                             pitch_model="pesto_crepevoicing", render_svg=False,
                             same_pitch_merge="auto" if phase_g_post else "off",
                             median_smooth_g5="auto" if phase_g_post else "off",
                             silent_trim_g6="auto" if phase_g_post else "off")
        r = transcribe(str(wav), cfg)
        notes = list(r.notes)
        if eval_sec is not None:
            notes = [n for n in notes if n.onset_s < eval_sec]
        # Humming branch: monophonic. G-1 leaves voice=0; G-2 still uses pred beats.
        voices = [0] * len(notes)
        beats_arr = None
        if use_beats and r.beats is not None and len(r.beats) >= 2:
            beats_arr = [b for b in r.beats if b < (eval_sec + 5.0 if eval_sec else 1e9)]
        pred_text = notes_to_mv2h_format(
            notes, bpm=float(r.bpm) if r.bpm else 120.0,
            time_sig="4/4", voices=voices,
            tatums_per_beat=args.tatums_per_beat, beats=beats_arr,
        )
        try:
            res = compute_mv2h(pred_text, gt_text, align="non_aligned")
        except Exception as e:
            print(f"skip {clip}: {e}"); continue
        row = {"clip": clip, **res.as_dict(),
               "n_notes_gt": len(gt_notes), "n_notes_pred": len(notes),
               "bpm_pred": float(r.bpm)}
        rows.append(row)
        print(f"voc  {clip:18s} mv2h={res.mv2h:.4f} v={res.voice:.3f} m={res.meter:.3f} mp={res.multi_pitch:.3f} val={res.value:.3f}")
    mean = {k: float(np.nanmean([r[k] for r in rows]))
            for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony")} if rows else {}
    return {"rows": rows, "mean": mean,
            "use_beats": use_beats, "use_voices": use_voices,
            "alignment": "non_aligned", "annotator": annotator}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["asap"],
                    choices=["asap", "maestro", "vocadito"])
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--piece", default=None, help="restrict to one piece")
    ap.add_argument("--mode",
                    choices=["baseline", "g1_voices", "g2_beats", "g1g2_both",
                             "post_only", "g1g2_post"],
                    default="g1g2_both",
                    help="post_only: G-4/5/6 post-processing without voices/beats. "
                         "g1g2_post: full Phase G Stage 1 emission + post.")
    ap.add_argument("--tatums-per-beat", type=int, default=4)
    ap.add_argument("--out", default=None, help="output json path; overrides default")
    ap.add_argument("--limit", type=int, default=0, help="if >0, limit Vocadito/MAESTRO to N clips")
    args = ap.parse_args()

    use_beats = args.mode in ("g2_beats", "g1g2_both", "g1g2_post")
    use_voices = args.mode in ("g1_voices", "g1g2_both", "g1g2_post")
    phase_g_post = args.mode in ("post_only", "g1g2_post")
    out = {"mode": args.mode, "use_beats": use_beats, "use_voices": use_voices,
           "phase_g_post": phase_g_post}
    t0 = time.time()
    if "asap" in args.datasets:
        out["asap"] = _run_asap(args, use_beats=use_beats, use_voices=use_voices)
    if "maestro" in args.datasets:
        out["maestro"] = _run_maestro(args, use_beats=use_beats, use_voices=use_voices,
                                       phase_g_post=phase_g_post)
    if "vocadito" in args.datasets:
        out["vocadito"] = _run_vocadito(args, use_beats=use_beats, use_voices=use_voices,
                                          phase_g_post=phase_g_post)
    out["wall_s"] = time.time() - t0
    out_path = (Path(args.out) if args.out
                else Path(f"reports/_metric_mv2h_phase_g_{args.mode}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    if "asap" in args.datasets and out["asap"]["rows"]:
        m = out["asap"]["mean"]
        print(f"ASAP mean: mv2h={m['mv2h']:.4f} v={m['voice']:.3f} m={m['meter']:.3f} mp={m['multi_pitch']:.3f} val={m['value']:.3f} h={m['harmony']:.3f}")
    if "maestro" in args.datasets and out["maestro"]["rows"]:
        m = out["maestro"]["mean"]
        print(f"MAESTRO mean: mv2h={m['mv2h']:.4f} v={m['voice']:.3f} m={m['meter']:.3f} mp={m['multi_pitch']:.3f} val={m['value']:.3f} h={m['harmony']:.3f}")
    if "vocadito" in args.datasets and out["vocadito"]["rows"]:
        m = out["vocadito"]["mean"]
        print(f"Vocadito mean: mv2h={m['mv2h']:.4f} v={m['voice']:.3f} m={m['meter']:.3f} mp={m['multi_pitch']:.3f} val={m['value']:.3f} h={m['harmony']:.3f}")


if __name__ == "__main__":
    main()
