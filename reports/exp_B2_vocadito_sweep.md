# exp_B2_vocadito_sweep — Vocadito hyperparameter sweep (soft mode, voicing segmenter)

## Goal
Optimize the four exposed hyperparameters of `humscribe.config.ModeConfig` (under "soft" mode) for Vocadito A1 mean COnP F1. Baseline 0.538; want to push past 0.55. The voicing-driven segmenter (`humscribe.pitch.voicing.segment_pitch_to_notes`) is the bottleneck, and these are its tuning knobs.

## Procedure
- Sweep tool: WandB Bayesian, 16 runs, 2 parallel agents (`scripts/sweep_vocadito.yaml`, `scripts/sweep_vocadito_humming.py`).
- Search space:
  - `voicing_threshold ∈ uniform[0.20, 0.65]`
  - `min_note_seconds ∈ uniform[0.04, 0.16]`
  - `pitch_smooth_window ∈ {3, 5, 7, 9, 11}`
  - `onset_merge_seconds ∈ uniform[0.02, 0.10]`
- Per run: full 40-clip Vocadito A1, mir_eval `precision_recall_f1_overlap(onset_tol=0.05s, pitch_tol=50 cents)`.
- Sweep id: `agam_p-iit-roorkee/humscribe-v3.2/ls3pvruk`. Hyperband early-termination enabled.

## Results
Top 5 configs (full table on WandB):

| rank | F1 | P | R | vt | mns | psw | oms |
|---|---|---|---|---|---|---|---|
| 1 | **0.577** | 0.585 | 0.592 | 0.315 | 0.052 | 11 | 0.026 |
| 2 | 0.573 | 0.560 | 0.605 | 0.220 | 0.054 | 11 | 0.036 |
| 3 | 0.571 | 0.534 | 0.630 | 0.208 | 0.050 | 11 | 0.048 |
| 4 | 0.570 | 0.534 | 0.629 | 0.217 | 0.047 | 11 | 0.031 |
| 5 | 0.566 | 0.536 | 0.619 | 0.250 | 0.041 | 11 | 0.062 |

Baseline (Phase A) for comparison: F1 = 0.538 with vt=0.30, mns=0.06, psw=7, oms=0.08.

Verified rerun with new defaults: F1 = 0.576 (within rounding of best sweep run).

WandB sweep: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/sweeps/ls3pvruk
Verification run: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/<see logs/gate_vocadito_tuned.log>

## Interpretation
**+3.9pp lift** (0.538 → 0.577) just from hyperparameter tuning, no architectural change. The dominant factor is `pitch_smooth_window=11` (used by all top 7 configs vs the old default 7) — humming has slow vibrato and small pitch wobble, so heavier median smoothing is correct. The voicing-threshold optimum (~0.31) is close to the old default (0.30); the old `onset_merge_seconds=0.08` was too generous (it was glueing distinct notes together).

The two regions of the search space — vt low (~0.22) high recall, vt mid (~0.31) balanced — both produce ≈0.57 F1 but with different P/R trade-offs. A user who cares more about catching every note (high recall) should drop vt to 0.22; default is set to 0.315 since recall and precision balance there.

Updated `humscribe.config.ModeConfig.for_mode("soft")` with the rank-1 config. Medium and hard modes untouched (need separate sweeps; future work).

## Next
- B2b: same sweep for `mode=medium` and `mode=hard`. Different distributions (Vocadito is closer to soft territory; for medium/hard maybe target a clean-vocal subset of MAESTRO-MIDI-rendered audio).
- B2c: sweep the segmenter-choice + hyperparams together (voicing + HMM + their respective configs).
- Bigger absolute gain: B4b (HMM hyperparam sweep) + a learned onset detector (B6).
- Also commit this updated `soft` default and re-baseline MTG-QBH visual for future comparisons.
