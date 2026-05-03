# exp_B45b_verify — HMM+hybrid on A1 + A2

## Goal
B45 found HMM+hybrid F1=0.671 (sigma_v=0.3, p_sustain=0.97, p_start=0.05) on Vocadito A1, +0.6pp over voicing-segmenter baseline 0.665. Verify on A2 to test cross-annotator robustness before promoting to default.

## Procedure
- Same B45 best HMM config.
- Vocadito A1 + A2, 40 clips each.
- Same hybrid voicing (PESTO pitch + CREPE periodicity).

## Results
| annotator | HMM+hybrid F1 | voicing+hybrid F1 (default) | Δ |
|---|---|---|---|
| A1 | **0.671** | 0.665 | +0.6pp |
| A2 | 0.626 | 0.630 | **-0.4pp** |
| mean | 0.649 | 0.648 | +0.1pp |

## Interpretation
HMM+hybrid wins narrowly on A1 but **regresses on A2**. Cross-annotator average is essentially tied. The HMM's tighter `sigma_voicing=0.3` is well-tuned for A1's voicing distribution but slightly mis-aligned for A2.

**Decision**: keep `voicing` segmenter as the default. The voicing-thresholded approach is more robust across annotators.

The `pitch_model="pesto_crepevoicing"` win remains the headline (+5-12pp), independent of which segmenter consumes the voicing signal.

## Final Vocadito numbers (locked)
- A1: 0.665 (voicing+hybrid)
- A2: 0.630 (voicing+hybrid)
- Cross-annotator mean: 0.648
- vs Phase A baselines: A1 +12.7pp, A2 +10.5pp

Phase B+1 truly complete. Future improvements need pre-trained features or substantially more training data.
