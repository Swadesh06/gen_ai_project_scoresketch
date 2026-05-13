"""Phase G G-9 strict measurement: per-note confidence vs is-in-GT correlation.

For each Vocadito clip:
- Run the pipeline with Phase G defaults (PESTO + CREPE + beat_this).
- aggregate_confidence() to populate `NoteEvent.confidence` from the
  pesto/crepe traces and beat positions.
- mir_eval-match predicted notes to GT (onset_tol=50ms, pitch_tol=50c).
- Each matched note gets `in_gt=1`, each unmatched (false positive)
  gets `in_gt=0`.

Outputs to `reports/_item-g9_per_note.json`:
- per-clip arrays of (confidence, in_gt)
- Pearson correlation across all notes
- Recall of false positives when flagging the lowest-20% conf
- Strict pass: |r| >= 0.4 AND lowest-20% recovers >= 60% of FPs
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import mir_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.config import PipelineConfig
from humscribe.eval.confidence import aggregate_confidence
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.post_process import median_smooth_pitch, merge_same_pitch, trim_silence

VOC = Path("/workspace/.cache/vocadito_orig")


def _load_gt(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in csv_path.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    hz = np.array([float(r[1]) for r in rows], dtype=np.float64)
    dur = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + dur], axis=1), hz


def _predict_with_confidence(wav: Path, *, phase_g_post: bool = True) -> list[NoteEvent]:
    cfg = PipelineConfig(input_kind="humming", mode="soft",
                         pitch_model="pesto_crepevoicing")
    audio, sr = load_audio(str(wav), target_sr=cfg.sample_rate)
    if phase_g_post:
        audio, _l, _t = trim_silence(audio, sr, db_threshold=cfg.silent_trim_db)
    t_p, hz_p, vp = track_pitch_pesto(audio, sr)
    t_c, _hz_c, vc = track_pitch_crepe(audio, sr)
    vc_on_p = np.interp(t_p, t_c, vc) if len(t_c) > 0 else vp
    hz_for_seg, vc_for_seg = hz_p, vc_on_p
    if phase_g_post:
        hz_for_seg, vc_for_seg = median_smooth_pitch(
            t_p, hz_for_seg, vc_for_seg, window_ms=cfg.median_smooth_window_ms,
        )
    notes = segment_pitch_to_notes(t_p, hz_for_seg, vc_for_seg, cfg.mode_config)
    notes = [n for n in notes if (n.offset_s - n.onset_s) >= cfg.mode_config.min_note_seconds]
    if phase_g_post:
        notes = merge_same_pitch(notes, gap_s=cfg.same_pitch_merge_ms / 1000.0)
    beats, _, _ = track_beats_beat_this(str(wav), target_bpm=110.0)
    aggregate_confidence(notes, (t_p, hz_p, vp), (t_c, _hz_c, vc), beats)
    return notes


def _label_in_gt(pred: Sequence[NoteEvent], gt_intervals: np.ndarray, gt_hz: np.ndarray,
                  onset_tol: float = 0.05, pitch_tol_cents: float = 50.0) -> list[int]:
    """For each predicted note, 1 if it has a GT match, else 0."""
    if not pred or len(gt_hz) == 0:
        return [0] * len(pred)
    est_iv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in pred], dtype=np.float64)
    est_hz = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in pred], dtype=np.float64)
    matching = mir_eval.transcription.match_notes(
        gt_intervals, gt_hz, est_iv, est_hz,
        onset_tolerance=onset_tol,
        pitch_tolerance=pitch_tol_cents,
        offset_ratio=None,
    )
    matched_est = {int(j) for (_i, j) in matching}
    return [1 if i in matched_est else 0 for i in range(len(pred))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--n-clips", type=int, default=0)
    ap.add_argument("--phase-g-post", choices=["on", "off"], default="on")
    ap.add_argument("--out", default="reports/_item-g9_per_note.json")
    args = ap.parse_args()
    phase_g = args.phase_g_post == "on"
    notes_dir = VOC / "Annotations" / "Notes"
    audio_dir = VOC / "Audio"
    pat = f"_notes{args.annotator}.csv"
    note_files = sorted(p for p in notes_dir.glob(f"*{pat}"))
    if args.n_clips > 0:
        note_files = note_files[: args.n_clips]
    all_conf: list[float] = []
    all_in_gt: list[int] = []
    per_clip = []
    for nf in note_files:
        clip_id = nf.stem.replace(f"_notes{args.annotator}", "")
        wav = audio_dir / f"{clip_id}.wav"
        if not wav.exists():
            continue
        gt_iv, gt_hz = _load_gt(nf)
        try:
            pred = _predict_with_confidence(wav, phase_g_post=phase_g)
        except Exception as e:
            print(f"skip {clip_id}: {e}")
            continue
        in_gt = _label_in_gt(pred, gt_iv, gt_hz)
        confs = [float(n.confidence) for n in pred]
        all_conf.extend(confs)
        all_in_gt.extend(in_gt)
        n_fp = sum(1 for x in in_gt if x == 0)
        per_clip.append({"clip": clip_id, "n_pred": len(pred), "n_fp": n_fp})
        print(f"{clip_id:18s} n_pred={len(pred):3d} fp={n_fp:3d}")
    conf_arr = np.array(all_conf, dtype=np.float64)
    in_gt_arr = np.array(all_in_gt, dtype=np.float64)
    pearson = float("nan")
    if conf_arr.size >= 3 and conf_arr.std() > 0 and in_gt_arr.std() > 0:
        from scipy.stats import pearsonr
        pearson = float(pearsonr(conf_arr, in_gt_arr)[0])
    # Lowest-20% flag: rank by conf ascending, take bottom 20% indices,
    # count how many of them are FPs (in_gt==0).
    n = len(conf_arr)
    flag_recall = float("nan")
    if n > 5:
        order = np.argsort(conf_arr)
        k = max(1, int(round(0.20 * n)))
        flagged = order[:k]
        flagged_fp = int(((1 - in_gt_arr[flagged]) > 0).sum())
        total_fp = int((1 - in_gt_arr).sum())
        flag_recall = float(flagged_fp / max(total_fp, 1))
    out = {
        "phase_g_post": args.phase_g_post,
        "annotator": args.annotator,
        "n_clips": len(per_clip),
        "n_notes": int(n),
        "n_fp": int((1 - in_gt_arr).sum()) if n else 0,
        "pearson_conf_vs_in_gt": pearson,
        "lowest_20pct_flag_recovers_fp_fraction": flag_recall,
        "per_clip": per_clip,
        "strict_pearson_pass": bool(abs(pearson) >= 0.4) if pearson == pearson else None,
        "strict_lowest_20pct_pass": bool(flag_recall >= 0.6) if flag_recall == flag_recall else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\npearson(conf, in_gt) = {pearson:+.3f}  (strict |r| >= 0.4)")
    print(f"lowest-20% flagged recovers {flag_recall:.1%} of FPs  (strict >= 60%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
