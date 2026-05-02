"""Debug Stage-5 quarterLength match. Tries multiple repair strategies."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import mir_eval
import music21
import numpy as np

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

ASAP = Path("/home/swadesh/datasets/asap")
PERF = "Bach/Fugue/bwv_846/Shi05M.mid"
WAV = ASAP / PERF.replace(".mid", ".wav")
SCORE_XML = ASAP / PERF.rsplit("/", 1)[0] / "xml_score.musicxml"
TPB = 12
ALLOWED_QL = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])


def snap_to_allowed(d: float) -> float:
    return float(ALLOWED_QL[np.argmin(np.abs(ALLOWED_QL - d))])


ann = json.loads((ASAP / "asap_annotations.json").read_text())
gt_beats = np.array(ann[PERF]["performance_beats"], dtype=np.float64)
notes = transcribe_piano(str(WAV))
onsets = np.array([n.onset_s for n in notes], dtype=np.float64)
offsets = np.array([n.offset_s for n in notes], dtype=np.float64)
print(f"pred notes: {len(notes)}  gt beats: {len(gt_beats)}")

q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, gt_beats, tatums_per_beat=TPB)
pred_durs = (q_off - q_on) / float(TPB)

score = music21.converter.parse(str(SCORE_XML))
gt_durations = [float(n.quarterLength) for n in score.flatten().notes]
gt_d = np.array(gt_durations, dtype=np.float64)
print(f"\ngt note count (no chord expansion, music21): {len(gt_d)}")

n_pairs = min(len(pred_durs), len(gt_d))
print(f"\n=== A. raw index-pairs ===")
m = int(np.sum(np.abs(pred_durs[:n_pairs] - gt_d[:n_pairs]) < 0.05))
print(f"  match: {m}/{n_pairs} = {100*m/n_pairs:.1f}%")

snapped = np.array([snap_to_allowed(float(d)) for d in pred_durs])
print(f"\n=== B. snap pred to allowed-set, then index-pair ===")
m = int(np.sum(np.abs(snapped[:n_pairs] - gt_d[:n_pairs]) < 0.05))
print(f"  match: {m}/{n_pairs} = {100*m/n_pairs:.1f}%")

next_onset_durs = np.empty(len(onsets), dtype=np.float64)
sort = np.argsort(onsets)
times_sorted = onsets[sort]
for i, idx in enumerate(sort):
    if i + 1 < len(sort):
        gap = times_sorted[i + 1] - times_sorted[i]
    else:
        gap = float(offsets[idx] - onsets[idx])
    next_onset_durs[idx] = max(gap, 0.01)
avg_beat = float(np.mean(np.diff(gt_beats)))
nx_q = next_onset_durs / avg_beat
nx_snap = np.array([snap_to_allowed(float(d)) for d in nx_q])
print(f"\n=== C. next-onset gap as duration, snap, index-pair ===")
m = int(np.sum(np.abs(nx_snap[:n_pairs] - gt_d[:n_pairs]) < 0.05))
print(f"  match: {m}/{n_pairs} = {100*m/n_pairs:.1f}%")

print(f"\n=== D. raw next-onset gap (no snap), index-pair ===")
m = int(np.sum(np.abs(nx_q[:n_pairs] - gt_d[:n_pairs]) < 0.05))
print(f"  match: {m}/{n_pairs} = {100*m/n_pairs:.1f}%")

print(f"\n=== E. onset-aligned matching, then snapped duration compare ===")
ref_iv: list[tuple[float, float]] = []
ref_p: list[float] = []
for n in score.flatten().notes:
    on = float(n.offset)
    off = on + float(n.quarterLength)
    if hasattr(n, "pitches"):
        for p in n.pitches:
            ref_iv.append((on, off))
            ref_p.append(float(p.frequency))
    else:
        ref_iv.append((on, off))
        ref_p.append(float(n.pitch.frequency))
ref_iv_a = np.array(ref_iv, dtype=np.float64)
ref_p_a = np.array(ref_p, dtype=np.float64)
print(f"  GT (with chord expansion): {len(ref_iv_a)} notes")

t0 = float(gt_beats[0])
est_on_q = (onsets - t0) / avg_beat
est_off_q = (offsets - t0) / avg_beat
mask = est_on_q >= 0
est_on_q = est_on_q[mask]
est_off_q = np.maximum(est_off_q[mask], est_on_q + 1e-3)
est_iv = np.column_stack([est_on_q, est_off_q])
est_p = np.array([n.pitch_hz if n.pitch_hz else 440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes], dtype=np.float64)[mask]
matched = mir_eval.transcription.match_notes(
    ref_iv_a, ref_p_a, est_iv, est_p,
    onset_tolerance=0.5, pitch_tolerance=50.0, offset_ratio=None,
)
print(f"  matched pairs: {len(matched)}/{len(est_iv)}")
if matched:
    pi = [m[1] for m in matched]
    gi = [m[0] for m in matched]
    pd = pred_durs[mask][pi]
    gd = (ref_iv_a[gi, 1] - ref_iv_a[gi, 0])
    raw_m = int(np.sum(np.abs(pd - gd) < 0.05))
    pd_s = np.array([snap_to_allowed(float(d)) for d in pd])
    snap_m = int(np.sum(np.abs(pd_s - gd) < 0.05))
    print(f"  aligned raw ql match: {raw_m}/{len(matched)} = {100*raw_m/len(matched):.1f}%")
    print(f"  aligned snapped ql match: {snap_m}/{len(matched)} = {100*snap_m/len(matched):.1f}%")
    nxd = nx_q[mask][pi]
    nxd_s = np.array([snap_to_allowed(float(d)) for d in nxd])
    nxraw = int(np.sum(np.abs(nxd - gd) < 0.05))
    nxs = int(np.sum(np.abs(nxd_s - gd) < 0.05))
    print(f"  aligned next-onset raw ql match: {nxraw}/{len(matched)} = {100*nxraw/len(matched):.1f}%")
    print(f"  aligned next-onset snapped ql match: {nxs}/{len(matched)} = {100*nxs/len(matched):.1f}%")
