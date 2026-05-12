"""Phase E item 1: MV2H end-to-end metric on 40 Vocadito clips.

Vocadito ships per-clip note annotations as CSV (onset_s, freq_hz, duration_s)
under Annotations/Notes/vocadito_N_notesA{1,2}.csv. We convert those to MV2H
format and compare against pipeline.transcribe() in humming mode.

Outputs:
- reports/_metric_mv2h_vocadito.json
- WandB run tagged phase-e, metric-mv2h, gate, vocadito
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.pipeline import transcribe


VOC = Path("/home/swadesh/datasets/vocadito")
OUT_JSON_DEFAULT = Path("reports/_metric_mv2h_vocadito.json")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return -1
    midi = 69 + 12 * np.log2(hz / 440.0)
    return int(round(midi))


def _load_voc_notes(csv_path: Path) -> list[NoteEvent]:
    out = []
    for line in csv_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            on = float(parts[0]); hz = float(parts[1]); dur = float(parts[2])
        except ValueError:
            continue
        midi = _hz_to_midi(hz)
        if midi < 1:
            continue
        out.append(NoteEvent(onset_s=on, offset_s=on + max(dur, 1e-3),
                              pitch_midi=midi, pitch_hz=hz, velocity=80))
    return out


def _gt_text(csv_path: Path, eval_seconds: float | None) -> tuple[str, int]:
    notes = _load_voc_notes(csv_path)
    if eval_seconds is not None:
        notes = [n for n in notes if n.onset_s < eval_seconds]
    # Vocadito clips have no time signature; assume 4/4. Tempo estimated from
    # median IOI: 60 / median_ioi gives BPM in beats per minute.
    if len(notes) >= 2:
        iois = np.diff([n.onset_s for n in notes])
        median_ioi = float(np.median(iois)) if len(iois) > 0 else 0.5
        bpm = 60.0 / max(median_ioi, 0.1)
    else:
        bpm = 120.0
    return (notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4",
                                  voices=[0] * len(notes)),
            len(notes))


def _pred_text(wav_path: Path, pitch_model: str, eval_seconds: float | None) -> tuple[str, int, float]:
    cfg = PipelineConfig(input_kind="humming", mode="soft",
                          pitch_model=pitch_model, render_svg=False)
    r = transcribe(str(wav_path), cfg)
    notes = list(r.notes)
    if eval_seconds is not None:
        notes = [n for n in notes if n.onset_s < eval_seconds]
    return (notes_to_mv2h_format(notes, bpm=float(r.bpm) if r.bpm else 120.0,
                                  time_sig="4/4", voices=[0] * len(notes)),
            len(notes), float(r.bpm))


def _wandb_init(args):
    try:
        import wandb
        return wandb.init(project="humscribe-v3.2",
                          name=f"eval_mv2h_vocadito_{args.annotator}_{args.pitch_model}",
                          tags=["phase-e", "metric-mv2h", "gate", "vocadito"],
                          config={"annotator": args.annotator,
                                  "pitch_model": args.pitch_model,
                                  "eval_seconds": args.eval_seconds,
                                  "alignment": args.align,
                                  "limit": args.limit},
                          dir="logs/wandb", reinit=False)
    except Exception as e:
        print(f"wandb disabled: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", choices=["A1", "A2"], default="A1")
    ap.add_argument("--pitch-model", choices=["pesto", "crepe", "pesto_crepevoicing"],
                    default="pesto_crepevoicing")
    ap.add_argument("--eval-seconds", type=float, default=0.0,
                    help="0 = full clip")
    ap.add_argument("--align", choices=["non_aligned", "aligned"], default="non_aligned")
    ap.add_argument("--limit", type=int, default=40,
                    help="how many clips (Vocadito has 40)")
    args = ap.parse_args()

    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    run = _wandb_init(args)
    rows = []; skipped = []
    t0 = time.time()

    wavs = sorted((VOC / "Audio").glob("vocadito_*.wav"),
                  key=lambda p: int(p.stem.split("_")[1]))[: args.limit]
    print(f"found {len(wavs)} clips")
    for wav in wavs:
        clip_id = wav.stem  # 'vocadito_1'
        ann = VOC / "Annotations" / "Notes" / f"{clip_id}_notes{args.annotator}.csv"
        if not ann.exists():
            skipped.append({"clip": clip_id, "reason": "no_annotation"})
            print(f"skip {clip_id}: no A{args.annotator} annotation"); continue
        try:
            gt_text, n_gt = _gt_text(ann, eval_sec)
            pred_text, n_pred, bpm_pred = _pred_text(wav, args.pitch_model, eval_sec)
        except Exception as e:
            skipped.append({"clip": clip_id, "reason": f"prep_failed: {e}"})
            print(f"skip {clip_id}: prep failed -- {e}"); continue
        try:
            res = compute_mv2h(pred_text, gt_text, align=args.align)
        except Exception as e:
            skipped.append({"clip": clip_id, "reason": f"mv2h_failed: {e}"})
            print(f"skip {clip_id}: mv2h -- {e}"); continue
        row = {"clip": clip_id, **res.as_dict(),
               "n_notes_gt": n_gt, "n_notes_pred": n_pred, "bpm_pred": bpm_pred}
        rows.append(row)
        print(f"{clip_id:14s} mv2h={res.mv2h:.4f}  mp={res.multi_pitch:.3f} "
              f"voice={res.voice:.3f} meter={res.meter:.3f} value={res.value:.3f} "
              f"harm={res.harmony:.3f} (gt={n_gt} pred={n_pred})")
        if run is not None:
            run.log({f"{clip_id}/mv2h": res.mv2h,
                     f"{clip_id}/mp": res.multi_pitch,
                     f"{clip_id}/voice": res.voice,
                     f"{clip_id}/meter": res.meter,
                     f"{clip_id}/value": res.value,
                     f"{clip_id}/harmony": res.harmony})
    mean = {}
    if rows:
        for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony"):
            mean[k] = float(np.nanmean([r[k] for r in rows]))
        print("---")
        print(f"mean MV2H = {mean['mv2h']:.4f}  "
              f"mp={mean['multi_pitch']:.3f} voice={mean['voice']:.3f} "
              f"meter={mean['meter']:.3f} value={mean['value']:.3f} "
              f"harmony={mean['harmony']:.3f}")
        if run is not None:
            run.log({f"mean/{k}": v for k, v in mean.items()})
    # Suffix the output by annotator so A1 and A2 don't overwrite each other.
    out_json = OUT_JSON_DEFAULT.with_name(
        OUT_JSON_DEFAULT.stem + f"_{args.annotator}" + OUT_JSON_DEFAULT.suffix
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "annotator": args.annotator, "pitch_model": args.pitch_model,
        "alignment": args.align, "eval_seconds": eval_sec,
        "rows": rows, "mean": mean, "skipped": skipped,
        "wall_s": time.time() - t0,
    }, indent=2))
    # Also keep the canonical A1 location for backwards compat.
    if args.annotator == "A1":
        OUT_JSON_DEFAULT.write_text(out_json.read_text())
    print(f"wrote {out_json}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
