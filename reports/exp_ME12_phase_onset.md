# ME-12 — phase-deviation onset detector

## Goal

Phase E item 7 ME-12 per `task_description_v3.md`. Pure-DSP onset
detector (librosa's `onset_detect` on a median-aggregated mel-spectrogram
with phase-deviation features). Different feature space than the
voicing-based onset estimator → uncorrelated vote.

## Procedure

`humscribe/ensemble/me12_phase_onset.py`: thin wrapper around
`librosa.onset.onset_detect` with `feature=melspectrogram` and
`aggregate=np.median`. No training, no model — pure CPU.

`scripts/eval_me12_phase_onset.py`: compares per-clip onset F1 against
Vocadito A1 GT onsets at ±50ms tolerance.

## Results

**Mean onset F1 = 0.538** across all 40 Vocadito clips.

Per-clip range: 0.42–0.64. The detector finds many onsets but some
are spurious peaks (notably consonant attacks vs vibrato modulations).

| metric | mean | range |
|---|---|---|
| onset F1 (event-level, ±50ms) | 0.538 | 0.42–0.64 |
| precision | ~0.50 | 0.42–0.66 |
| recall | ~0.55 | 0.38–0.65 |

**Comparison context**: this is onset-event F1 (no pitch requirement),
not directly comparable to Vocadito's note-level noff F1 = 0.665 (which
requires onset AND pitch agreement). The detector's value is as a
**second uncorrelated onset estimator** to vote alongside the voicing-
based path.

## Interpretation

ME-12 alone doesn't beat any existing onset method. But its uncorrelated
errors (vibrato-driven false positives are different from voicing-dip
false negatives) make it a useful **vote** when used in an ensemble:

- Voicing path: catches vibrato-stable notes; misses notes that begin
  with a soft attack.
- Phase-deviation: catches soft attacks; emits false positives on
  vibrato cycles.

Their union (with deduplication) could pick up notes the voicing missed
while suppressing each method's specific FP modes. Concrete integration:
require both to fire within 30 ms for the onset to be accepted.

## Decision

Implementation kept (`humscribe.ensemble.me12_phase_onset`). Phase F-2
combined with the formant offset detector forms a natural pair:
phase-deviation for onsets, formant-band BiLSTM for offsets, both
operating on the same 80-bin mel-spec.

Not promoted to production default in this session — needs the
voting-fusion logic (item 7 ME-1's pattern, redone with safer voicing
arithmetic).

## Files

- `humscribe/ensemble/me12_phase_onset.py`
- `scripts/eval_me12_phase_onset.py`
- `reports/_exp_ME12_phase_onset.json`
