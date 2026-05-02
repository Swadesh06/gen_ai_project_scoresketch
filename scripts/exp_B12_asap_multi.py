"""Run ASAP Stage 4+5 over multiple pieces; aggregate metrics."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mir_eval
import music21
import numpy as np
import pretty_midi
import wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import (
    adaptive_tatums_per_beat, viterbi_quantize_rhythm,
)

ASAP = Path("~/datasets/asap").expanduser()
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])


def render_midi(midi: Path, wav: Path) -> None:
    if wav.exists() and wav.stat().st_size > 0:
        return
    cmd = ["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)]
    subprocess.run(cmd, check=True, capture_output=True)


def load_score_beats(ann: Path) -> np.ndarray:
    beats = []
    for line in ann.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                beats.append(float(parts[0]))
            except ValueError:
                continue
    return np.array(sorted(beats), dtype=np.float64)


def load_midi_notes(mid: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv: list[list[float]] = []; pi: list[float] = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv, dtype=np.float64), np.array(pi, dtype=np.float64)


def snap(d: float) -> float:
    return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


def pick_pieces(n: int) -> list[Path]:
    """Pick n piano-piece dirs that have midi_score.mid + matching annotations."""
    candidates = sorted(ASAP.rglob("midi_score.mid"))
    chosen: list[Path] = []
    for m in candidates:
        d = m.parent
        if (d / "midi_score_annotations.txt").exists() and (d / "xml_score.musicxml").exists():
            chosen.append(d)
        if len(chosen) >= n:
            break
    return chosen


def run_piece(piece_dir: Path, beat_tol: float = 0.07) -> dict:
    score_mid = piece_dir / "midi_score.mid"
    score_wav = piece_dir / "midi_score.wav"
    score_ann = piece_dir / "midi_score_annotations.txt"
    render_midi(score_mid, score_wav)
    beats = load_score_beats(score_ann)
    avg_beat = float(np.diff(beats).mean()) if len(beats) >= 2 else 0.5
    pred_beats, _, bpm = track_beats_beat_this(str(score_wav))
    f_beat = float(mir_eval.beat.f_measure(beats, pred_beats, f_measure_threshold=beat_tol))
    notes = transcribe_piano(str(score_wav))
    onsets = np.array([n.onset_s for n in notes], dtype=np.float64)
    offsets = np.array([n.offset_s for n in notes], dtype=np.float64)
    tpb = 24
    q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, beats, tatums_per_beat=tpb)
    pred_durs = (q_off - q_on) / float(tpb)
    gt_iv, gt_p = load_midi_notes(score_mid)
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes], dtype=np.float64)
    matched = mir_eval.transcription.match_notes(
        gt_iv, gt_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    pct_raw = pct_snap = 0.0
    if matched:
        gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
        gd = gt_durs_q[gi]; pd = pred_durs[pi]
        pct_raw = float(np.mean(np.abs(pd - gd) < 0.05))
        gd_s = np.array([snap(float(x)) for x in gd])
        pd_s = np.array([snap(float(x)) for x in pd])
        pct_snap = float(np.mean(pd_s == gd_s))
    return {
        "piece": str(piece_dir.relative_to(ASAP)),
        "n_gt_beats": int(len(beats)),
        "n_pred_beats": int(len(pred_beats)),
        "f_beat": f_beat,
        "n_pred_notes": int(len(notes)),
        "n_gt_notes": int(len(gt_iv)),
        "n_matched": int(len(matched)),
        "stage5_raw": pct_raw,
        "stage5_snap": pct_snap,
        "bpm": float(bpm),
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(n_pieces: int) -> None:
    pieces = pick_pieces(n_pieces)
    cfg = {"exp": "B12_asap_multi", "n_pieces": len(pieces), "git_sha": git_sha(),
            "tatums_per_beat": 24}
    run = wandb.init(
        project="humscribe-v3.2", name=f"exp_B12_asap_multi_n{len(pieces)}",
        config=cfg, tags=["B12", "asap", "multi"], dir="logs/wandb",
    )
    rows: list[dict] = []
    for i, p in enumerate(pieces):
        print(f"\n[{i+1}/{len(pieces)}] {p.relative_to(ASAP)}")
        try:
            r = run_piece(p)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        rows.append(r)
        print(f"  beat F={r['f_beat']:.3f}  raw={r['stage5_raw']:.3f}  snap={r['stage5_snap']:.3f}  bpm={r['bpm']:.0f}  notes={r['n_pred_notes']}/{r['n_gt_notes']}")
        wandb.log({"piece_idx": i, **{f"piece/{k}": v for k, v in r.items() if isinstance(v, (int, float))}})
    if rows:
        f_beats = [r["f_beat"] for r in rows]
        raws = [r["stage5_raw"] for r in rows]
        snaps = [r["stage5_snap"] for r in rows]
        summary = {
            "n_pieces": len(rows),
            "mean_f_beat": float(np.mean(f_beats)),
            "median_f_beat": float(np.median(f_beats)),
            "mean_stage5_raw": float(np.mean(raws)),
            "mean_stage5_snap": float(np.mean(snaps)),
            "median_stage5_snap": float(np.median(snaps)),
            "stage4_pct_pass_90": float(np.mean([f > 0.90 for f in f_beats])),
            "stage5_pct_pass_60": float(np.mean([s >= 0.60 for s in snaps])),
        }
        wandb.summary.update(summary)
        print(f"\nSummary over {len(rows)} pieces:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    out = Path(f"reports/_exp_B12_asap_multi.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary if rows else {}}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pieces", type=int, default=5)
    args = ap.parse_args()
    main(args.n_pieces)
