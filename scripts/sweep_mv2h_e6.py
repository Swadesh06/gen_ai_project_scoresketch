"""Phase E item 6: MV2H-driven hyperparameter sweep agent.

Reads cached features from /workspace/.cache/sweep_e6_features/, runs only
the rhythm-DP + render + MV2H path for one config of (TPB, complexity_alpha,
sigma_quant, voicing_psw, voicing_vt, target_bpm, allowed_denoms), and logs
the per-piece + mean MV2H to WandB. Designed to be launched as ~6 parallel
agents under a WandB sweep YAML.

Use:
    wandb sweep scripts/sweep_mv2h_e6.yaml
    wandb agent <SWEEP_ID>                  # one terminal
    wandb agent <SWEEP_ID>                  # another terminal (× 6)

Or run a single config standalone:
    python scripts/sweep_mv2h_e6.py
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.config import ModeConfig
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

CACHE = Path("/workspace/.cache/sweep_e6_features")

ASAP_KEYS = [
    "Bach__Fugue__bwv_854",
    "Beethoven__Piano_Sonatas__21-1",
    "Chopin__Berceuse_op_57",
    "Liszt__Sonata",
    "Schumann__Toccata",
]
VOC_IDS = list(range(1, 11))


def _eval_asap_piece(key: str, *, tpb: int, complexity_alpha: float,
                      sigma_quant: float, target_bpm: float,
                      dp_offgrid_penalty: float) -> tuple[float, dict] | None:
    npz = CACHE / f"asap_{key}.npz"
    gt = CACHE / f"asap_{key}_gt.txt"
    if not npz.exists() or not gt.exists():
        return None
    d = np.load(npz)
    on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
    beats = d["beats"]
    if len(beats) < 2 or len(on) == 0:
        return None
    # Run rhythm DP and re-snap onsets/offsets to the predicted tatum grid.
    q_on, q_off = viterbi_quantize_rhythm(
        on, off, beats, tatums_per_beat=int(tpb),
        offgrid_penalty=float(dp_offgrid_penalty),
    )
    bpm = float(d["bpm"][0])
    if bpm <= 0:
        bpm = 120.0
    tatum_s = 60.0 / (bpm * float(tpb))
    # First note's quantised onset anchors the tatum 0 origin in seconds.
    on_origin = float(on[0]) - q_on[0] * tatum_s if len(on) > 0 else 0.0
    notes = []
    for i in range(len(on)):
        m = int(midi[i])
        if m < 1 or m > 127: continue
        new_on = on_origin + int(q_on[i]) * tatum_s
        new_off = on_origin + int(q_off[i]) * tatum_s
        if new_off <= new_on:
            new_off = new_on + tatum_s
        notes.append(NoteEvent(onset_s=new_on, offset_s=new_off,
                               pitch_midi=m, velocity=80))
    pred_text = notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4",
                                      voices=[0]*len(notes))
    gt_text = gt.read_text()
    res = compute_mv2h(pred_text, gt_text, align="non_aligned", timeout_s=60.0)
    return res.mv2h, res.as_dict()


def _eval_voc_clip(vid: int, annotator: str, *,
                    tpb: int, sigma_quant: float, target_bpm: float,
                    voicing_psw: int, voicing_vt: float,
                    dp_offgrid_penalty: float) -> tuple[float, dict] | None:
    npz = CACHE / f"voc_{vid}.npz"
    gt = CACHE / f"voc_{vid}_{annotator}_gt.txt"
    if not npz.exists() or not gt.exists():
        return None
    d = np.load(npz)
    t = d["t"]; hz = d["hz"]; vc = d["vc"]
    beats = d["beats"]
    mc = ModeConfig(voicing_threshold=float(voicing_vt),
                     min_note_seconds=0.052,
                     onset_merge_seconds=0.026,
                     dp_offgrid_penalty=float(dp_offgrid_penalty),
                     pitch_smooth_window=int(voicing_psw))
    notes = segment_pitch_to_notes(t, hz, vc, mc)
    if not notes:
        return None
    bpm = float(d["bpm"][0])
    if bpm <= 0: bpm = 120.0
    on = np.array([n.onset_s for n in notes], dtype=np.float64)
    off = np.array([n.offset_s for n in notes], dtype=np.float64)
    if len(beats) >= 2 and len(on) > 0:
        q_on, q_off = viterbi_quantize_rhythm(
            on, off, beats, tatums_per_beat=int(tpb),
            offgrid_penalty=float(dp_offgrid_penalty),
        )
        tatum_s = 60.0 / (bpm * float(tpb))
        on_origin = float(on[0]) - q_on[0] * tatum_s
        new_notes = []
        for i, n in enumerate(notes):
            new_on = on_origin + int(q_on[i]) * tatum_s
            new_off = on_origin + int(q_off[i]) * tatum_s
            if new_off <= new_on:
                new_off = new_on + tatum_s
            new_notes.append(NoteEvent(onset_s=new_on, offset_s=new_off,
                                        pitch_midi=n.midi(),
                                        velocity=n.velocity))
        notes = new_notes
    pred_text = notes_to_mv2h_format(notes, bpm=bpm, time_sig="4/4",
                                      voices=[0]*len(notes))
    gt_text = gt.read_text()
    res = compute_mv2h(pred_text, gt_text, align="non_aligned", timeout_s=60.0)
    return res.mv2h, res.as_dict()


def _wandb_init(cfg: dict):
    try:
        import wandb
        return wandb.init(project="humscribe-v3.2",
                          name=f"sweep_e6_tpb{cfg['tpb']}_a{cfg['complexity_alpha']:.2f}",
                          tags=["phase-e", "sweep", "metric-mv2h"],
                          config=cfg, dir="logs/wandb", reinit=False)
    except Exception as e:
        print(f"wandb disabled: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--tpb", type=int, default=24)
    # Accept both dash and underscore forms so WandB sweep's --foo_bar args
    # work alongside our CLI's --foo-bar.
    ap.add_argument("--complexity-alpha", "--complexity_alpha",
                    dest="complexity_alpha", type=float, default=1.0)
    ap.add_argument("--sigma-quant", "--sigma_quant",
                    dest="sigma_quant", type=float, default=0.03)
    ap.add_argument("--voicing-psw", "--voicing_psw",
                    dest="voicing_psw", type=int, default=19)
    ap.add_argument("--voicing-vt", "--voicing_vt",
                    dest="voicing_vt", type=float, default=0.75)
    ap.add_argument("--target-bpm", "--target_bpm",
                    dest="target_bpm", type=float, default=110.0)
    ap.add_argument("--dp-offgrid-penalty", "--dp_offgrid_penalty",
                    dest="dp_offgrid_penalty", type=float, default=0.5)
    ap.add_argument("--annotator", choices=["A1", "A2"], default="A1")
    ap.add_argument("--out-json", "--out_json", dest="out_json",
                    type=Path, default=None)
    args = ap.parse_args()

    cfg = vars(args).copy(); cfg.pop("out_json", None)
    run = _wandb_init(cfg)
    rows = []; t0 = time.time()

    for key in ASAP_KEYS:
        try:
            r = _eval_asap_piece(
                key, tpb=args.tpb,
                complexity_alpha=args.complexity_alpha,
                sigma_quant=args.sigma_quant,
                target_bpm=args.target_bpm,
                dp_offgrid_penalty=args.dp_offgrid_penalty,
            )
        except Exception as e:
            print(f"asap {key} err: {e}"); continue
        if r is None: continue
        mv2h, details = r
        rows.append({"piece": key, "kind": "asap", **details})
        print(f"asap {key:42s} mv2h={mv2h:.4f}")
        if run is not None:
            run.log({f"asap/{key}/mv2h": mv2h})

    for vid in VOC_IDS:
        try:
            r = _eval_voc_clip(
                vid, args.annotator, tpb=args.tpb,
                sigma_quant=args.sigma_quant,
                target_bpm=args.target_bpm,
                voicing_psw=args.voicing_psw,
                voicing_vt=args.voicing_vt,
                dp_offgrid_penalty=args.dp_offgrid_penalty,
            )
        except Exception as e:
            print(f"voc_{vid} err: {e}"); continue
        if r is None: continue
        mv2h, details = r
        rows.append({"piece": f"voc_{vid}", "kind": "voc", **details})
        print(f"voc_{vid:2d}                                  mv2h={mv2h:.4f}")
        if run is not None:
            run.log({f"voc/{vid}/mv2h": mv2h})

    if rows:
        asap_mean = float(np.nanmean([r["mv2h"] for r in rows if r["kind"]=="asap"]))
        voc_mean = float(np.nanmean([r["mv2h"] for r in rows if r["kind"]=="voc"]))
        overall = float(np.nanmean([r["mv2h"] for r in rows]))
        print(f"---\nmean asap mv2h = {asap_mean:.4f}  "
              f"voc mv2h = {voc_mean:.4f}  overall = {overall:.4f}  "
              f"wall = {time.time()-t0:.1f}s")
        if run is not None:
            run.summary["asap_mean_mv2h"] = asap_mean
            run.summary["voc_mean_mv2h"] = voc_mean
            run.summary["overall_mv2h"] = overall
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({"cfg": cfg, "rows": rows}, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
