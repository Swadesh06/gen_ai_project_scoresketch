# exp_B19..B25_batch — parallel batch (B19, B20, B21, B22, B23, B25)

Six experiments launched simultaneously across GPU + CPU after the user requested aggressive parallelism.

## Results summary

| exp | metric | result | vs baseline | status |
|---|---|---|---|---|
| B19 mel-BiLSTM 5-fold CV | Vocadito mean F1 | 0.562 | -3.5pp vs voicing 0.597 | discard |
| B20 HMM voice tracker (single piece) | ASAP BWV 846 snap | 0.825 | -2.2pp vs greedy 0.847 | discard |
| B21 HMM VT hyperparam sweep | best snap | 0.825 | structural ceiling, same as default | discard |
| **B22 Vocadito psw extreme range** | A1 F1 | **0.597** | **+2.1pp vs psw=11** | **keep** |
| B23 DP hyperparam sweep on BWV 854 | snap | 0.904 | unchanged (already at optimum) | informative |
| B25 ASAP multi-piece w/ HMM VT | mean snap | 0.845 | -1.1pp vs greedy 0.856 | discard |

## B22 details — the one win

`pitch_smooth_window` extended from {3,5,7,9,11} (B2 sweep range) to {3,5,...,25}:

| psw | Vocadito A1 mean F1 |
|---|---|
| 3 | 0.529 |
| 5 | 0.537 |
| 7 | 0.549 |
| 9 | 0.561 |
| 11 (B2 default) | 0.576 |
| 13 | 0.590 |
| **15** | **0.597** |
| 17 | 0.596 |
| 19 | 0.591 |
| 21 | 0.583 |
| 25 | 0.565 |

Monotone in psw up to 15, then declines. The B2 sweep capped at 11 — extending the range finds a clear better optimum. Updated `ModeConfig.for_mode("soft")` default to psw=15.

Confirmed on Vocadito A2 (`exp_B28`): F1 0.525 → 0.551 (+2.6pp). Cross-annotator generalization holds.

## Why so many losses

- **B19 mel-BiLSTM**: 5-fold CV gives a more honest val number than B10's single split, but training set is still tiny (~30 clips per fold). Mel features should help with more data — adding MAESTRO-rendered humming or a synthesised augmentation set would unlock this. Phase B+1.
- **B20/B21/B25 HMM voice tracker**: The HMM has higher per-step expressiveness (probabilistic vs greedy) but in practice, the greedy assigner with `pitch_jump=3, time_gap_s=0.5` (B16 defaults) over-fragments into ~70 voices on Bach Fugue, which gives the per-voice DP much cleaner per-voice timings than HMM's principled ~8 voices. The HMM is "right" in theory but loses on this dataset because pitch lines in Bach can cross within 8-voice budget.
- **B23 DP sweep**: BWV 854 was already at 0.90 — there's no room. A sweep on harder pieces (BWV 856 at 0.808) would be more diagnostic.

## Cumulative

Vocadito A1 trajectory: **0.538 (Phase A) → 0.577 (B2) → 0.597 (B22)**. Total +5.9pp.

ASAP BWV 846 Stage-5 snap unchanged: **0.847** (best with greedy VT + B16 defaults, no improvement from this batch).
