"""Phase E item 1: MV2H end-to-end metric on the 5 MAESTRO chamber clips.

Audio + GT MIDI pairs live in `outputs/maestro_clips/*.{wav,mid}`. We run
`humscribe.pipeline.transcribe()` (instrument path with current production
defaults) on each WAV and compare the resulting notes to the GT MIDI.

Outputs:
- reports/_metric_mv2h_maestro.json
- WandB run tagged phase-e, metric-mv2h, gate, maestro
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.pipeline import transcribe


CLIPS_DIR = Path("outputs/maestro_clips")
OUT_JSON = Path("reports/_metric_mv2h_maestro.json")


def _gt_text(mid_path: Path, eval_seconds: float | None) -> tuple[str, int]:
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
    return notes_to_mv2h_format(notes, bpm=bpm, time_sig=ts, voices=voices), len(notes)


def _pred_text(wav_path: Path, eval_seconds: float | None,
                transcriber: str = "bytedance_piano") -> tuple[str, int, float]:
    cfg = PipelineConfig(input_kind="piano", mode="medium", render_svg=False,
                          per_voice_dp="off", transcriber=transcriber)
    r = transcribe(str(wav_path), cfg)
    notes = list(r.notes)
    if eval_seconds is not None:
        notes = [n for n in notes if n.onset_s < eval_seconds]
    return (notes_to_mv2h_format(notes, bpm=float(r.bpm) if r.bpm else 120.0,
                                  time_sig="4/4", voices=[0] * len(notes)),
            len(notes), float(r.bpm))


def _wandb_init(args: argparse.Namespace):
    try:
        import wandb
        return wandb.init(project="humscribe-v3.2",
                          name="eval_mv2h_maestro",
                          tags=["phase-e", "metric-mv2h", "gate", "maestro"],
                          config={"eval_seconds": args.eval_seconds,
                                  "alignment": args.align},
                          dir="logs/wandb", reinit=False)
    except Exception as e:
        print(f"wandb disabled: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--align", choices=["non_aligned", "aligned"], default="non_aligned")
    ap.add_argument("--transcriber", choices=["bytedance_piano", "yourmt3plus", "auto_piano"],
                    default="bytedance_piano",
                    help="bytedance_piano is fast (~3 GB VRAM); yourmt3plus is the "
                         "production default but ~5 GB and slow first-load.")
    args = ap.parse_args()

    eval_sec = None if args.eval_seconds <= 0 else float(args.eval_seconds)
    run = _wandb_init(args)
    rows = []
    skipped = []
    t0 = time.time()

    wavs = sorted(CLIPS_DIR.glob("*.wav"))
    print(f"found {len(wavs)} MAESTRO clips in {CLIPS_DIR}")
    for wav in wavs:
        mid = wav.with_suffix(".mid")
        if not mid.exists():
            skipped.append({"clip": wav.name, "reason": "no_gt"}); continue
        try:
            gt_text, n_gt = _gt_text(mid, eval_sec)
            pred_text, n_pred, bpm_pred = _pred_text(wav, eval_sec, args.transcriber)
        except Exception as e:
            skipped.append({"clip": wav.name, "reason": f"prep_failed: {e}"})
            print(f"skip {wav.name}: {e}")
            continue
        try:
            res = compute_mv2h(pred_text, gt_text, align=args.align,
                                jar_dir=Path("third_party/MV2H/bin").resolve())
        except Exception as e:
            skipped.append({"clip": wav.name, "reason": f"mv2h_failed: {e}"})
            print(f"skip {wav.name}: {e}")
            continue
        row = {"clip": wav.name, **res.as_dict(),
               "n_notes_gt": n_gt, "n_notes_pred": n_pred, "bpm_pred": bpm_pred}
        rows.append(row)
        print(f"{wav.name:60s} mv2h={res.mv2h:.4f}  mp={res.multi_pitch:.3f} "
              f"voice={res.voice:.3f} meter={res.meter:.3f} value={res.value:.3f} "
              f"harm={res.harmony:.3f} (gt={n_gt}, pred={n_pred})")
        if run is not None:
            run.log({f"{wav.stem}/mv2h": res.mv2h,
                     f"{wav.stem}/mp": res.multi_pitch,
                     f"{wav.stem}/voice": res.voice,
                     f"{wav.stem}/meter": res.meter,
                     f"{wav.stem}/value": res.value,
                     f"{wav.stem}/harmony": res.harmony})
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
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "alignment": args.align, "eval_seconds": eval_sec,
        "rows": rows, "mean": mean, "skipped": skipped,
        "wall_s": time.time() - t0,
    }, indent=2))
    print(f"wrote {OUT_JSON}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
