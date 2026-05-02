"""Sweep voice tracker hyperparameters on cached ByteDance output.

The expensive parts (ByteDance + beat_this) are computed once; only the VT +
DP run per (pitch_jump, time_gap_s) point. Fast — finishes in seconds total.
"""
from __future__ import annotations
import json
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, assign_voices, per_voice_durations

ROOT = Path("/home/swadesh/datasets/asap/Bach/Fugue/bwv_846")
TPB = 24
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])


def snap(d: float) -> float:
    return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


def main() -> None:
    score_wav = ROOT / "midi_score.wav"
    score_mid = ROOT / "midi_score.mid"
    score_ann = ROOT / "midi_score_annotations.txt"

    score_beats = np.array(sorted(float(line.split()[0]) for line in score_ann.read_text().splitlines() if line.split()))
    avg_beat = float(np.diff(score_beats).mean())
    pm = pretty_midi.PrettyMIDI(str(score_mid))
    gt_iv = []; gt_p = []
    for inst in pm.instruments:
        for n in inst.notes:
            gt_iv.append([n.start, n.end])
            gt_p.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    gt_iv = np.array(gt_iv); gt_p = np.array(gt_p)
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat

    print("running ByteDance once ...")
    notes = transcribe_piano(str(score_wav))
    onsets_orig = np.array([n.onset_s for n in notes], dtype=np.float64)
    offsets_orig = np.array([n.offset_s for n in notes], dtype=np.float64)
    est_iv = np.column_stack([onsets_orig, np.maximum(offsets_orig, onsets_orig + 1e-3)])
    est_p = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes], dtype=np.float64)
    matched = mir_eval.transcription.match_notes(
        gt_iv, gt_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    gi = [m[0] for m in matched]; pi_idx = [m[1] for m in matched]
    print(f"matched: {len(matched)}/{len(gt_iv)}")

    run = wandb.init(project="humscribe-v3.2", name="exp_B16_vt_sweep",
                     tags=["B16", "voice_tracking", "sweep"], dir="logs/wandb")
    rows: list[dict] = []
    for pj in (3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 12.0):
        for tg in (0.5, 1.0, 1.5, 2.0, 3.0):
            cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=tg)
            voices = assign_voices(notes, cfg)
            on_v, off_v = per_voice_durations(notes, voices)
            q_on, q_off = viterbi_quantize_rhythm(
                on_v, off_v, score_beats, tatums_per_beat=TPB,
                allowed_durations_tatums=default_allowed_durations(TPB),
            )
            pred_durs = (q_off - q_on) / float(TPB)
            pd = pred_durs[pi_idx]; gd = gt_durs_q[gi]
            raw = float(np.mean(np.abs(pd - gd) < 0.05))
            pd_s = np.array([snap(float(x)) for x in pd])
            gd_s = np.array([snap(float(x)) for x in gd])
            snap_pct = float(np.mean(pd_s == gd_s))
            n_voices = len(voices)
            rows.append({"pitch_jump": pj, "time_gap": tg, "raw": raw, "snap": snap_pct, "n_voices": n_voices})
            print(f"  pj={pj:4.1f}  tg={tg:.1f}  raw={raw:.3f}  snap={snap_pct:.3f}  n_voices={n_voices}")
            wandb.log({"pj": pj, "tg": tg, "raw": raw, "snap": snap_pct, "n_voices": n_voices})

    rows.sort(key=lambda r: -r["snap"])
    print(f"\nTop 5 by snap:")
    for r in rows[:5]:
        print(f"  snap={r['snap']:.3f}  raw={r['raw']:.3f}  pj={r['pitch_jump']}  tg={r['time_gap']}  voices={r['n_voices']}")
    out = Path("reports/_exp_B16_vt_sweep.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    main()
