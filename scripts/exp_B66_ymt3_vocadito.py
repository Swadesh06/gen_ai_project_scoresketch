"""B66 — YourMT3+ end-to-end on Vocadito (Phase C, speculative).

YMT3+ was trained on multi-instrument data including some vocal sources.
Test whether it does anything useful on monophonic humming. If yes, we can
add it as a third route in `auto_humming` — but the current humming path
(PESTO + CREPE + voicing → DP) already hits 0.665 vs IAA ceiling 0.740,
so the bar is high.

Score against A1 + A2 with mir_eval no_offset / offset50 / offset20.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.yourmt3plus import transcribe_yourmt3plus
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

VOC = Path("~/datasets/vocadito").expanduser()
OUT_JSON = Path("reports/_exp_B66_ymt3_vocadito.json")
CACHE = Path("/workspace/.cache/vocadito_ymt3")
CACHE.mkdir(parents=True, exist_ok=True)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_notes(annotation_csv: Path):
    """Vocadito CSV: onset_s, pitch_hz, duration_s."""
    rows = [r.strip().split(",") for r in annotation_csv.read_text().splitlines() if r.strip()]
    if not rows:
        return np.empty((0, 2)), np.empty(0)
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    intervals = np.stack([onsets, onsets + durations], axis=1)
    return intervals, pitches


def score(pred_intervals, pred_pitches, ref_intervals, ref_pitches, offset):
    if len(pred_intervals) == 0 or len(ref_intervals) == 0:
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, pred_intervals, pred_pitches,
        offset_ratio=offset, onset_tolerance=0.05,
    )[2]


def transcribe_cached(wav_path: Path):
    cache = CACHE / f"{wav_path.stem}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["intervals"], d["pitches"]
    notes = transcribe_yourmt3plus(str(wav_path))
    if not notes:
        intervals = np.empty((0, 2), dtype=np.float64)
        pitches = np.empty(0, dtype=np.float64)
    else:
        intervals = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)]
                               for n in notes], dtype=np.float64)
        # NoteEvent.midi is a method
        pitches = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes],
                            dtype=np.float64)
    np.savez(cache, intervals=intervals, pitches=pitches)
    return intervals, pitches


def main(n_clips: int = 40) -> None:
    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    clips = sorted(p.stem for p in audio_dir.glob("vocadito_*.wav"))[:n_clips]
    cfg_w = {"git_sha": git_sha(), "n_clips": len(clips)}
    run = wandb.init(project="humscribe-v3.2", name="exp_B66_ymt3_vocadito",
                     config=cfg_w, tags=["B66", "vocadito", "yourmt3", "phase-c"],
                     dir="logs/wandb")
    rows = []
    for clip in clips:
        wav = audio_dir / f"{clip}.wav"
        a1 = ann_dir / f"{clip}_notesA1.csv"
        a2 = ann_dir / f"{clip}_notesA2.csv"
        if not a1.exists():
            continue
        try:
            iv, pi = transcribe_cached(wav)
        except Exception as e:
            print(f"  {clip} predict failed: {e}")
            continue
        ref1_iv, p1_hz = load_notes(a1)
        out = {"clip": clip, "n_pred": int(len(iv)), "n_ref": int(len(ref1_iv))}
        for off_label, off_val in [("noff", None), ("o50", 0.5), ("o20", 0.2)]:
            out[f"{off_label}_a1"] = score(iv, pi, ref1_iv, p1_hz, off_val)
        if a2.exists():
            ref2_iv, p2_hz = load_notes(a2)
            for off_label, off_val in [("noff", None), ("o50", 0.5), ("o20", 0.2)]:
                out[f"{off_label}_a2"] = score(iv, pi, ref2_iv, p2_hz, off_val)
                out[f"{off_label}_soft"] = 0.5 * (out[f"{off_label}_a1"] + out[f"{off_label}_a2"])
        rows.append(out)
        print(f"  {clip}: n={len(iv)} a1.noff={out['noff_a1']:.3f}", end="")
        if "noff_a2" in out:
            print(f" a2={out['noff_a2']:.3f} soft={out['noff_soft']:.3f}")
        else:
            print()
    if not rows:
        print("no clips scored")
        run.finish(); return
    keys = [k for k in rows[0].keys() if k.startswith(("noff_", "o50_", "o20_"))]
    means = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    means["mean_n_pred"] = float(np.mean([r["n_pred"] for r in rows]))
    means["mean_n_ref"] = float(np.mean([r["n_ref"] for r in rows]))
    print("\nMEANS:")
    for k, v in means.items():
        print(f"  {k:12s} = {v:.4f}")
    wandb.summary.update(means)
    OUT_JSON.write_text(json.dumps({"rows": rows, "means": means, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
