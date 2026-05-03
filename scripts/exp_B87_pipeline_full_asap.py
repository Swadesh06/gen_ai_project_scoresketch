"""B87 — Full pipeline.transcribe() vs B63 baseline on the 9-piece ASAP test set.

B63 used the old greedy + shared-DP path with YMT3+ transcription and got
9-piece mean snap = 0.774. B87 runs the new pipeline.transcribe() with
auto-routing (per_voice_dp routes Chopin to B76 transformer + per-voice DP)
and reports the headline number.

Per-piece comparison + mean delta. The expected change is small because
only Chopin should re-route (per the heuristic in pipeline._should_use_per_voice_dp).
"""
from __future__ import annotations
import json
import pickle
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import wandb

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe, _should_use_per_voice_dp
from humscribe.notes import NoteEvent

ASAP = Path("~/datasets/asap").expanduser()
RENDER = Path("/workspace/.cache/asap_renders")
OUT_JSON = Path("reports/_exp_B87_pipeline_full_asap.json")
TPB = 24
ALLOWED_BEATS = np.array([0.0625, 0.083, 0.125, 0.167, 0.1875, 0.25, 0.333,
                            0.375, 0.5, 0.667, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

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


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_score_notes(piece: str):
    mid = pretty_midi.PrettyMIDI(str(ASAP / piece / "midi_score.mid"))
    notes = []
    for inst in mid.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort(key=lambda x: x[0])
    intervals = np.array([[n[0], n[1]] for n in notes])
    pitches = np.array([440.0 * 2 ** ((n[2] - 69) / 12) for n in notes])
    return intervals, pitches


def snap_quantize(d: float) -> float:
    return float(ALLOWED_BEATS[np.argmin(np.abs(ALLOWED_BEATS - d))])


def compute_snap(notes, q_on, q_off, beats, gt_iv, gt_pi):
    """B63 snap metric: fraction of matched notes whose pred dur quantizes to GT dur."""
    if not notes or len(beats) < 2:
        return {"snap": 0.0, "n_matched": 0}
    avg_beat = float(np.mean(np.diff(beats)))
    pred_durs = (q_off - q_on) / TPB  # in beats
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_pi = np.array([n.pitch_hz if n.pitch_hz else
                        (440.0 * 2 ** ((n.midi() - 69) / 12)) for n in notes])
    matched = mir_eval.transcription.match_notes(
        gt_iv, gt_pi, est_iv, est_pi,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    if not matched:
        return {"snap": 0.0, "n_matched": 0}
    pd = pred_durs[[m[1] for m in matched]]
    gd = (gt_iv[[m[0] for m in matched], 1] - gt_iv[[m[0] for m in matched], 0]) / avg_beat
    pd_s = np.array([snap_quantize(float(x)) for x in pd])
    gd_s = np.array([snap_quantize(float(x)) for x in gd])
    return {"snap": float(np.mean(pd_s == gd_s)), "n_matched": len(matched)}


def main() -> None:
    cfg_w = {"git_sha": git_sha(), "n_pieces": len(PIECES)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B87_pipeline_full_asap",
                     config=cfg_w, tags=["B87", "asap", "integration", "phase-d"],
                     dir="logs/wandb")

    rows = []
    for piece, safe in PIECES:
        wav = RENDER / f"{safe}.wav"
        if not wav.exists():
            print(f"  skip {piece}: no rendered audio at {wav}")
            continue
        print(f"\n=== {piece} ===")
        # Use cfg.transcriber=yourmt3plus (the new default) + per_voice_dp=auto
        cfg = PipelineConfig(input_kind="piano", per_voice_dp="auto",
                              transcriber="yourmt3plus", render_svg=False)
        try:
            res = transcribe(str(wav), cfg)
        except Exception as e:
            print(f"  transcribe failed: {e}")
            continue
        print(f"  pipeline output: {res.n_notes} notes, bpm={res.bpm:.1f}")
        # Did auto-route fire?
        was_pvd = _should_use_per_voice_dp(res.notes, cfg)
        print(f"  per_voice_dp auto-route: {was_pvd}")

        gt_iv, gt_pi = load_score_notes(piece)
        snap_res = compute_snap(res.notes, res.tatum_onsets, res.tatum_offsets,
                                  res.beats, gt_iv, gt_pi)
        print(f"  snap (B63 metric): {snap_res['snap']:.4f}  matched={snap_res['n_matched']}")
        rows.append({"piece": piece, "n_notes": res.n_notes,
                      "auto_route_pvd": was_pvd,
                      "snap_b87": snap_res["snap"],
                      "n_matched": snap_res["n_matched"]})
        wandb.log({"piece": piece, **rows[-1]})

    if not rows:
        run.finish(); return
    means = {
        "mean_snap_5bach": float(np.mean([r["snap_b87"] for r in rows
                                              if r["piece"].startswith("Bach")])),
        "mean_snap_4romantic": float(np.mean([r["snap_b87"] for r in rows
                                                  if not r["piece"].startswith("Bach")])),
        "mean_snap_9overall": float(np.mean([r["snap_b87"] for r in rows])),
    }
    print(f"\nMEANS over {len(rows)} pieces:")
    for k, v in means.items():
        print(f"  {k:24s} = {v:.4f}")
    print("\nvs B63 baseline (greedy + shared DP, same YMT3+ transcription):")
    print(f"  9-piece mean: B63=0.774  B87={means['mean_snap_9overall']:.4f}  delta={means['mean_snap_9overall']-0.774:+.4f}")
    wandb.summary.update(means)
    OUT_JSON.write_text(json.dumps({"rows": rows, "means": means, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
