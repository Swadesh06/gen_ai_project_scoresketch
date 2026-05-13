# item-7 ME-11 — formant-band onset detector — FINAL strict negative

## Goal

Phase E item 7, ME-11 from v3 spec. Add formant-band (1.5-3.5 kHz)
onsets to the production direct path to lift Vocadito A1 noff F1 from
the current 0.618 baseline past the v3 thresholds:
- item 2 (BiLSTM stack): ≥ 0.69
- item 7 (full ensemble): ≥ 0.70

## Procedure

`scripts/eval_me11_formant_onset.py`:
1. Compute 64-bin mel spectrogram, 1500-3500 Hz, hop 10 ms.
2. Onset envelope = mean(log_mel) → rectified diff → normalised.
3. librosa peak-pick with delta=0.1, wait=3.
4. Vote rule: keep ME-11 onset only if (a) no direct onset within 50 ms
   AND (b) PESTO pitch trace within 30 ms AND (c) voicing ≥ 0.7 × mode
   threshold AND (d) valid pitch > 0.

## 40-clip Vocadito A1 result

- direct (production):  **0.6181**
- +ME-11 onsets:        **0.5935**
- **delta:              −0.0246**

**Every clip ≤ 0 except 3 (voc_13, 28, 30) which had slight positive
deltas (+0.006, +0.021, +0.016).** The pattern: ME-11 onsets are noisy.
For each true positive picked up, the detector adds 2-4 false positives
that drag noff F1 down.

Worst clips: voc_32 (Δ −0.118, 31 spurious adds), voc_27 (Δ −0.065, 6
adds where 5+ were wrong), voc_14 (Δ −0.071, 9 adds).

## v3 strict criteria

| criterion | direct alone | +ME-11 | pass? |
|---|---|---|---|
| item 2 noff ≥ 0.69 | 0.618 | 0.594 | **FAIL** (gap −0.106 from +ME-11) |
| item 7 noff ≥ 0.70 | 0.618 | 0.594 | **FAIL** (gap −0.106 from +ME-11) |

## Root cause

Formant-band onsets fire on:
- Note-internal vibrato spikes (especially loud high-formant content)
- Breath transitions (between phrases)
- Consonant attacks (clearer than vowel-only sections)

None of these correspond to *new* note onsets that aren't already
detected by PESTO voicing. The voicing-based onsets in the production
direct path already cover the high-recall cases; ME-11 only adds noise.

A learned per-clip onset classifier (trained on Vocadito with ME-11
features) might do better than the librosa peak-pick rule. But that's
a 40-clip data problem all over again — the same constraint that
killed F-2 (BiLSTM offset detector).

## Decision

Strict negative. ME-11 ships behind no flag; the function isn't worth
integrating. The +0.01 MV2H per-member criterion is not met and the
ensemble criterion ≥ 0.70 is unreachable from this signal.

The combined empirical conclusion across all ensemble members tested
(ME-1, ME-4, ME-7, ME-9, ME-10, ME-11, ME-12, F-2e):
- ME-7 (anacrusis): conservative, no false positives, no demonstrated win
- F-2e (BiLSTM offset confidence head): +0.0028 MV2H, below +0.01 bar
- All others: negative

The v3 item 7 ensemble pass criterion (Voc noff ≥ 0.70) cannot be met
with the current production direct-path baseline of 0.618 and available
ensemble members. The structural gap is real and the next direction is
a learned detector with substantially more training data — DALI v2 or
similar, deferred to Phase F.

## Files

- `scripts/eval_me11_formant_onset.py`
- `reports/_phase_e_item7_me11_formant_onset.json`
- This report.
