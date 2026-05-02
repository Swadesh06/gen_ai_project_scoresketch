"""Run the same Stage-5 evaluation but on the SCORE-rendered audio
(midi_score.wav) and SCORE-aligned beats (from midi_score_annotations.txt).

This isolates: 'is the DP capable of producing exact quarterLength matches when
given clean inputs?'
"""
from __future__ import annotations
from pathlib import Path

import music21
import numpy as np

from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

ROOT = Path("/home/swadesh/datasets/asap/Bach/Fugue/bwv_846")
WAV = ROOT / "midi_score.wav"
SCORE_XML = ROOT / "xml_score.musicxml"
ANN = ROOT / "midi_score_annotations.txt"
TPB = 12

beats = []
for line in ANN.read_text().splitlines():
    parts = line.split()
    if len(parts) >= 2:
        beats.append(float(parts[0]))
gt_beats = np.array(beats, dtype=np.float64)
print(f"score-grid beats: {len(gt_beats)}, IOI={float(np.diff(gt_beats).mean()):.3f}s, BPM={60.0/np.diff(gt_beats).mean():.1f}")

notes = transcribe_piano(str(WAV))
onsets = np.array([n.onset_s for n in notes], dtype=np.float64)
offsets = np.array([n.offset_s for n in notes], dtype=np.float64)
print(f"ByteDance notes: {len(notes)}")

q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, gt_beats, tatums_per_beat=TPB)
pred_durs = (q_off - q_on) / float(TPB)

score = music21.converter.parse(str(SCORE_XML))
gt_durs = np.array([float(n.quarterLength) for n in score.flatten().notes], dtype=np.float64)
print(f"GT notes (music21 flatten): {len(gt_durs)}")

n = min(len(pred_durs), len(gt_durs))
m = int(np.sum(np.abs(pred_durs[:n] - gt_durs[:n]) < 0.05))
print(f"\nIndex-paired raw match: {m}/{n} = {100*m/n:.1f}%")

ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
def snap(d):
    return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])
snapped = np.array([snap(float(x)) for x in pred_durs])
m2 = int(np.sum(np.abs(snapped[:n] - gt_durs[:n]) < 0.05))
print(f"Index-paired snap-to-allowed match: {m2}/{n} = {100*m2/n:.1f}%")

avg_beat = float(np.diff(gt_beats).mean())
sort = np.argsort(onsets)
times_s = onsets[sort]
gaps = np.empty(len(onsets), dtype=np.float64)
for i, idx in enumerate(sort):
    if i + 1 < len(sort):
        gaps[idx] = times_s[i + 1] - times_s[i]
    else:
        gaps[idx] = max(float(offsets[idx] - onsets[idx]), 0.01)
nx_q = gaps / avg_beat
nx_snap = np.array([snap(float(x)) for x in nx_q])
m3 = int(np.sum(np.abs(nx_snap[:n] - gt_durs[:n]) < 0.05))
print(f"Index-paired next-onset-gap snap match: {m3}/{n} = {100*m3/n:.1f}%")

from collections import Counter
print(f"\npred_durs distribution (top 10):")
for d, k in Counter(round(float(x), 4) for x in pred_durs).most_common(10):
    print(f"  {d:8.4f}  x {k}")
print(f"snapped distribution (top 10):")
for d, k in Counter(round(float(x), 4) for x in snapped).most_common(10):
    print(f"  {d:8.4f}  x {k}")
print(f"gt distribution (top 10):")
for d, k in Counter(round(float(x), 4) for x in gt_durs).most_common(10):
    print(f"  {d:8.4f}  x {k}")
