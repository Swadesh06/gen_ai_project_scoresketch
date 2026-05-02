"""ASAP multi-piece on DIVERSE composers (not just Bach Fugues). Tests
generalization of the B15+B16 voice-tracking + DP defaults beyond the homogeneous
Bach 4-voice fugue distribution."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.voice_tracking import quantize_with_voice_tracking

ASAP = Path("~/datasets/asap").expanduser()
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])


def render_midi(midi: Path, wav: Path) -> None:
    if wav.exists() and wav.stat().st_size > 0:
        return
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)], check=True, capture_output=True)


def load_score_beats(ann: Path) -> np.ndarray:
    beats = []
    for line in ann.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try: beats.append(float(parts[0]))
            except ValueError: continue
    return np.array(sorted(beats), dtype=np.float64)


def load_midi_notes(mid: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv = []; pi = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv, dtype=np.float64), np.array(pi, dtype=np.float64)


def snap(d): return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


PICKED = [
    "Bach/Fugue/bwv_846", "Mozart/Piano_Sonatas/13-2", "Chopin/Berceuse_op_57",
    "Beethoven/Piano_Sonatas/21-1", "Schumann/Toccata", "Liszt/Sonata",
]


def pick_diverse_pieces(n: int) -> list[Path]:
    chosen = []
    for k in PICKED:
        d = ASAP / k
        if (d / "midi_score.mid").exists() and (d / "midi_score_annotations.txt").exists():
            chosen.append(d)
        if len(chosen) >= n:
            break
    return chosen


def run_piece(piece_dir: Path) -> dict:
    mid = piece_dir / "midi_score.mid"
    wav = piece_dir / "midi_score.wav"
    ann = piece_dir / "midi_score_annotations.txt"
    render_midi(mid, wav)
    beats = load_score_beats(ann)
    avg_beat = float(np.diff(beats).mean()) if len(beats) >= 2 else 0.5
    pred_beats, _, bpm = track_beats_beat_this(str(wav), target_bpm=60.0/avg_beat)
    f_beat = float(mir_eval.beat.f_measure(beats, pred_beats, f_measure_threshold=0.07))
    notes = transcribe_piano(str(wav))
    onsets = np.array([n.onset_s for n in notes]); offsets = np.array([n.offset_s for n in notes])
    q_on, q_off = quantize_with_voice_tracking(notes, beats, tatums_per_beat=24)
    pred_durs = (q_off - q_on) / 24.0
    gt_iv, gt_p = load_midi_notes(mid)
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes])
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, est_p, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    pct_raw = pct_snap = 0.0
    if matched:
        gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
        gd = gt_durs_q[gi]; pd = pred_durs[pi]
        pct_raw = float(np.mean(np.abs(pd - gd) < 0.05))
        gd_s = np.array([snap(float(x)) for x in gd])
        pd_s = np.array([snap(float(x)) for x in pd])
        pct_snap = float(np.mean(pd_s == gd_s))
    return {"piece": str(piece_dir.relative_to(ASAP)),
            "f_beat": f_beat, "stage5_raw": pct_raw, "stage5_snap": pct_snap,
            "n_pred": int(len(notes)), "n_gt": int(len(gt_iv)),
            "n_matched": int(len(matched)), "bpm": float(bpm)}


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main(n_pieces: int) -> None:
    pieces = pick_diverse_pieces(n_pieces)
    cfg = {"exp": "B37_asap_diverse", "git_sha": git_sha(), "n_pieces": len(pieces)}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B37_asap_diverse_n{len(pieces)}",
                      config=cfg, tags=["B37", "asap", "diverse"], dir="logs/wandb")
    rows = []
    for i, p in enumerate(pieces):
        print(f"\n[{i+1}/{len(pieces)}] {p.relative_to(ASAP)}")
        try: r = run_piece(p)
        except Exception as e:
            print(f"  failed: {e}"); continue
        rows.append(r)
        print(f"  beat F={r['f_beat']:.3f}  raw={r['stage5_raw']:.3f}  snap={r['stage5_snap']:.3f}  bpm={r['bpm']:.0f}")
        wandb.log({"piece_idx": i, **{f"piece/{k}": v for k, v in r.items() if isinstance(v, (int, float))}})
    if rows:
        snaps = [r["stage5_snap"] for r in rows]
        f_beats = [r["f_beat"] for r in rows]
        summary = {"n": len(rows), "mean_f_beat": float(np.mean(f_beats)),
                   "mean_stage5_snap": float(np.mean(snaps)),
                   "median_stage5_snap": float(np.median(snaps)),
                   "stage5_pct_pass_85": float(np.mean([s >= 0.85 for s in snaps]))}
        wandb.summary.update(summary)
        print(f"\nSummary: mean snap={summary['mean_stage5_snap']:.3f}  beat_F={summary['mean_f_beat']:.3f}  pass85={summary['stage5_pct_pass_85']*100:.0f}%")
    out = Path("reports/_exp_B37_asap_diverse.json")
    out.write_text(json.dumps({"rows": rows, "summary": summary if rows else {}}, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pieces", type=int, default=6)
    main(**vars(ap.parse_args()))
