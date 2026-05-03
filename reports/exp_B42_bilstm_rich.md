# exp_B42_bilstm_rich — mel + PESTO + CREPE BiLSTM (5-fold CV)

## Goal
Re-do B19 with richer per-frame features:
- 32 log-mel band-energy features
- PESTO normalized midi
- PESTO confidence
- CREPE normalized midi
- CREPE periodicity
- (PESTO_midi - CREPE_midi) disagreement
- mean(PESTO_conf, CREPE_per)

37 input dims total. BiLSTM hidden=192, 2 layers, dropout 0.25, 60 epochs, AdamW lr=1e-3.

5-fold CV on the union of Vocadito A1 + A2 (40 clips × 2 = 80 examples; folds split by clip id so each clip's A1+A2 appear together in same fold).

## Results

| fold | val F1 |
|---|---|
| 1 | 0.582 |
| 2 | 0.662 |
| 3 | 0.620 |
| 4 | 0.605 |
| 5 | 0.626 |
| **mean** | **0.619** |

Voicing+hybrid baseline: 0.665. **BiLSTM loses by -4.6pp**.

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/1e0p20o2

## Interpretation
Adding mel + dual pitch features did improve over B10/B19 (0.490, 0.562) but still well below the heuristic. The 5-fold CV variance is large (0.582-0.662 range, 8pp), confirming the model is under-trained for the test variance.

The fundamental data limit: 80 examples × ~30s audio = ~40 min of training data. Modern onset detectors are trained on hundreds of hours.

Decision: discard. Document for future agents that the BiLSTM path needs much more data. Pre-trained features (MERT/MusicFM) should be the next attempt — they would inject the equivalent of millions of pre-training tokens.

## Final BiLSTM trajectory
| variant | F1 |
|---|---|
| B10 plain (PESTO features only, single split) | 0.490 |
| B19 mel-only (5-fold) | 0.562 |
| **B42b mel + PESTO + CREPE rich (5-fold)** | **0.619** |
| voicing+hybrid heuristic (no training) | **0.665** |

The heuristic (PESTO+CREPE-voicing+median-filter+threshold) wins by 4.6-17.5pp across all BiLSTM variants tried. Current Vocadito ceiling for any non-pre-trained approach.
