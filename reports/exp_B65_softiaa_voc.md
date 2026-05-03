# exp_B65 — Soft-IAA Vocadito scoring (Phase C)

## Goal
B51 found the Vocadito IAA ceiling at 0.740 (no_offset) by computing F1 of
A1 against A2 (or vice versa). The standard practice has been to score
each prediction against A1 OR A2 separately. Soft-IAA averages F1 across
both annotators per clip, giving a more honest single number that doesn't
over-fit to one annotator's idiosyncrasies.

This is a metric change only — no model or pipeline changes.

## Procedure

`scripts/exp_B65_softiaa_voc.py` runs the same humming pipeline as
`gate_vocadito_conp.py` (PESTO pitch + CREPE periodicity-as-voicing,
mode=soft, no DP — see "DP bypass" note below) on all 40 Vocadito clips,
scoring each prediction against both A1 and A2 with mir_eval at three
offset settings: `noff` (no offset check, COnP), `o50` (offset_ratio=0.5),
`o20` (offset_ratio=0.2).

Output for each (clip, offset_setting) tuple: `f1_a1`, `f1_a2`, and
`f1_soft = 0.5 * (f1_a1 + f1_a2)`.

### DP bypass
First-attempt B65 ran predictions through `pipeline.transcribe()`, which
applies Stage-5 DP rhythm quantization for *all* inputs including humming.
The DP shifted onsets by tens of ms, which collapsed mir_eval's 50ms
onset-tolerance to zero matches → all clips scored 0.000. Fix: do
pitch+voicing+min-note-filter directly (the same recipe `gate_vocadito_conp`
uses) and bypass DP on humming for scoring purposes. The score-rendering
pipeline still uses DP, but evaluation against absolute-time annotations
must compare the un-quantized note onsets.

## Results

| metric | A1 | A2 | **soft (mean of both)** |
|---|---|---|---|
| no-offset F1 | 0.6651 | 0.6281 | **0.6466** |
| offset50 F1 | 0.5733 | 0.5354 | **0.5543** |
| offset20 F1 | 0.4394 | 0.3997 | **0.4195** |

WandB run: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/hwt717e8

### Vs IAA ceiling

| offset | pipeline soft | IAA ceiling (B51) | gap |
|---|---|---|---|
| no-offset | 0.6466 | 0.740 | -9.3pp |
| offset20 | 0.4195 | 0.642 | -22.3pp |

The no-offset gap (-9.3pp) is the COnP gap to human agreement. The
offset20 gap (-22.3pp) is the duration-precision gap — the dominant unfixed
weakness on the humming side.

### Sanity vs Phase A baselines
- A1 noff: B65 = 0.6651, prior gate = 0.665 → match (within 0.001).
- A2 noff: B65 = 0.6281, prior gate = 0.630 → match (within 0.002).
- Reproducibility: confirmed.

## Interpretation
- Soft-IAA is a strictly better single number to report than A1-only or
  A2-only — it doesn't reward over-fitting to one annotator. The
  arithmetic mean is 0.6466.
- The soft-IAA gap to ceiling stays at ~9pp on no-offset and ~22pp on
  offset20. Same diagnostic conclusion as B51, but now expressed in a
  metric that's resilient to which annotator we picked.
- Adopting soft-IAA as a reporting headline would have been valuable
  earlier — it would have neutralised the small "A1 vs A2" debates around
  the Phase B sweep ablations.

## Decision
Promote `f1_soft` as the **headline Vocadito metric** for any future
report. Keep A1 and A2 numbers as breakdown — they tell us when one
annotator's style trumps the other.

## Status
keep — pure metric change, zero pipeline impact, lower-variance reporting.
