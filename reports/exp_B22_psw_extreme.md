# exp_B22_psw_extreme — Vocadito psw extended range

## Goal
B2 sweep capped `pitch_smooth_window` at 11. Test whether higher values keep climbing.

## Procedure
Single-axis sweep with all other B2-tuned params fixed. psw ∈ {3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25}. Vocadito A1 mean F1.

## Results
| psw | mean F1 |
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

Mode is at psw=15 with F1=0.597. Confirmed on Vocadito A2: 0.525 → 0.551 (+2.6pp).

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/h82k3uys

## Interpretation
The B2 sweep's prior on psw was {3, 5, 7, 9, 11} — extending one more octave doubles the win. Median-filter window of 15 frames at 10 ms hop = 150 ms smoothing — about the duration of a typical hummed note onset transient. This filters out vibrato wobble that the smaller window leaves jittering enough to fragment notes.

Updated `ModeConfig.for_mode("soft").pitch_smooth_window = 15`.

Followup B26 sweep (vt × mns × oms with psw=15) found no further improvement — the new ceiling for the PESTO+voicing-segmenter pipeline on Vocadito A1 is ~0.597.

## Next
B26 confirmed the 0.597 ceiling. To push higher, need:
- A learned segmenter trained on more data (Vocadito alone is too small — see B10/B19).
- Or a different pitch tracker calibration (CREPE+psw=15 = 0.586, B27, still loses).
