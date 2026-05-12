"""Phase E item 1: MV2H end-to-end metric on the 9-piece ASAP test set.

Two prediction sources:
- cached YMT3 outputs in /workspace/.cache/asap_yourmt3plus/<piece>.pkl
- the full Phase D pipeline (PipelineConfig) on the rendered audio in
  /workspace/.cache/asap_renders/<piece>.wav

Ground truth comes from the ASAP repo's `midi_score.mid` files, sparse-cloned
into /workspace/.cache/asap_score_midis/asap-repo/.

Outputs:
- reports/_metric_mv2h_asap.json with per-piece per-axis MV2H scores
- WandB run tagged `phase-e`, `metric-mv2h`, `gate`
- Correlation analysis vs cached snap/note-F1 baselines (item 1 pass criterion).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.eval.mv2h import compute_mv2h, MV2HResult, _DEFAULT_JAR_DIR
from humscribe.eval.mv2h_io import notes_to_mv2h_format, midi_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.pipeline import transcribe

CACHE_YMT3 = Path("/workspace/.cache/asap_yourmt3plus")
CACHE_RENDERS = Path("/workspace/.cache/asap_renders")
ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
OUT_JSON = Path("reports/_metric_mv2h_asap.json")

PIECES = [
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


def _ymt3_notes_to_pipeline(notes: list[dict]) -> list[NoteEvent]:
    """Cached YMT3 outputs are dicts with on/off/midi/vel/conf."""
    out: list[NoteEvent] = []
    for n in notes:
        midi = int(n["midi"])
        if midi < 1 or midi > 127:
            continue
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        out.append(NoteEvent(
            onset_s=float(n["on"]), offset_s=float(n["off"]),
            pitch_midi=midi, pitch_hz=hz,
            velocity=int(n.get("vel", 80)), confidence=float(n.get("conf", 1.0)),
        ))
    out.sort(key=lambda e: e.onset_s)
    return out


def _trim_to_seconds(notes: Iterable[NoteEvent], max_seconds: float) -> list[NoteEvent]:
    return [n for n in notes if n.onset_s < max_seconds]


def _bpm_estimate(beats: np.ndarray | None) -> float:
    if beats is None or len(beats) < 2:
        return 120.0
    ibis = np.diff(beats)
    median_ibi = float(np.median(ibis))
    return 60.0 / max(median_ibi, 1e-3)


def _gt_for_piece(piece_dir: str, eval_seconds: float | None) -> tuple[str, dict] | None:
    """Load GT MIDI for a piece; return (mv2h_text, info_dict) or None if missing."""
    mid_path = ASAP_REPO / piece_dir / "midi_score.mid"
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
    txt = notes_to_mv2h_format(notes, bpm=bpm, time_sig=ts, voices=voices)
    return txt, {"bpm": bpm, "time_sig": ts, "n_notes_gt": len(notes)}


def _ymt3_prediction_for_piece(piece_key: str, eval_seconds: float | None) -> tuple[str, dict] | None:
    pkl = CACHE_YMT3 / f"{piece_key}.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        cache = pickle.load(f)
    bpm = float(cache.get("bpm", 120.0))
    notes = _ymt3_notes_to_pipeline(cache["notes"])
    if eval_seconds is not None:
        notes = _trim_to_seconds(notes, eval_seconds)
    txt = notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4", voices=[0] * len(notes))
    return txt, {"bpm": bpm, "n_notes_pred": len(notes), "source": "ymt3_cache"}


def _pipeline_prediction_for_piece(piece_key: str, eval_seconds: float | None) -> tuple[str, dict] | None:
    audio = CACHE_RENDERS / f"{piece_key}.wav"
    if not audio.exists():
        return None
    cfg = PipelineConfig(input_kind="piano", mode="medium")
    r = transcribe(str(audio), cfg)
    notes = list(r.notes)
    if eval_seconds is not None:
        notes = _trim_to_seconds(notes, eval_seconds)
    txt = notes_to_mv2h_format(notes, bpm=float(r.bpm) if r.bpm else 120.0,
                               time_sig="4/4", voices=[0] * len(notes))
    return txt, {"bpm": float(r.bpm), "n_notes_pred": len(notes),
                  "source": "pipeline_full"}


def _wandb_init(args: argparse.Namespace):
    try:
        import wandb
        return wandb.init(project="humscribe-v3.2",
                          name=f"eval_mv2h_asap_{args.source}",
                          tags=["phase-e", "metric-mv2h", "gate", "asap"],
                          config={"source": args.source,
                                  "eval_seconds": args.eval_seconds,
                                  "alignment": args.align},
                          dir="logs/wandb", reinit=False)
    except Exception as e:
        print(f"wandb disabled: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ymt3_cache", "pipeline_full"],
                    default="ymt3_cache",
                    help="ymt3_cache: read cached YMT3 outputs (fast). "
                         "pipeline_full: re-run pipeline.transcribe() on audio (GPU).")
    ap.add_argument("--eval-seconds", type=float, default=30.0,
                    help="Trim both pred and GT to first N seconds to make scoring "
                         "tractable. Set to 0 for no trim (full piece).")
    ap.add_argument("--align", choices=["non_aligned", "aligned"],
                    default="non_aligned",
                    help="non_aligned: DTW (recommended for tempo-different "
                         "pred/GT pairs). aligned: only when timings share base.")
    ap.add_argument("--alignment-penalty", type=float, default=1.0)
    args = ap.parse_args()

    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    run = _wandb_init(args)
    rows = []
    skipped = []
    t0 = time.time()
    for piece_dir, piece_key in PIECES:
        gt = _gt_for_piece(piece_dir, eval_sec)
        if gt is None:
            skipped.append({"piece": piece_key, "reason": "no_gt"})
            print(f"skip {piece_key}: no GT MIDI")
            continue
        gt_text, gt_info = gt

        if args.source == "ymt3_cache":
            pred = _ymt3_prediction_for_piece(piece_key, eval_sec)
        else:
            pred = _pipeline_prediction_for_piece(piece_key, eval_sec)
        if pred is None:
            skipped.append({"piece": piece_key, "reason": "no_pred"})
            print(f"skip {piece_key}: no prediction")
            continue
        pred_text, pred_info = pred

        try:
            res = compute_mv2h(pred_text, gt_text,
                                align=args.align,
                                alignment_penalty=args.alignment_penalty)
        except Exception as e:
            skipped.append({"piece": piece_key, "reason": f"mv2h_exception: {e}"})
            print(f"skip {piece_key}: {e}")
            continue
        row = {"piece": piece_key, **res.as_dict(),
               "n_notes_gt": gt_info["n_notes_gt"],
               "n_notes_pred": pred_info["n_notes_pred"]}
        rows.append(row)
        print(f"{piece_key:42s} mv2h={res.mv2h:.4f}  "
              f"mp={res.multi_pitch:.3f} voice={res.voice:.3f} "
              f"meter={res.meter:.3f} value={res.value:.3f} "
              f"harm={res.harmony:.3f} "
              f"(gt={gt_info['n_notes_gt']} pred={pred_info['n_notes_pred']})")
        if run is not None:
            run.log({f"{piece_key}/mv2h": res.mv2h,
                     f"{piece_key}/mp": res.multi_pitch,
                     f"{piece_key}/voice": res.voice,
                     f"{piece_key}/meter": res.meter,
                     f"{piece_key}/value": res.value,
                     f"{piece_key}/harmony": res.harmony})

    mean = {}
    if rows:
        for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony"):
            mean[k] = float(np.nanmean([r[k] for r in rows]))
        print("---")
        print(f"mean MV2H = {mean['mv2h']:.4f}  "
              f"mp={mean['multi_pitch']:.3f}  voice={mean['voice']:.3f}  "
              f"meter={mean['meter']:.3f}  value={mean['value']:.3f}  "
              f"harmony={mean['harmony']:.3f}")
        if run is not None:
            run.log({f"mean/{k}": v for k, v in mean.items()})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "source": args.source, "alignment": args.align,
        "alignment_penalty": args.alignment_penalty,
        "eval_seconds": eval_sec,
        "rows": rows, "mean": mean, "skipped": skipped,
        "wall_s": time.time() - t0,
    }, indent=2))
    print(f"wrote {OUT_JSON}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
