# exp_B52 — BiLSTM with HuBERT features

## Goal
Replace mel + PESTO/CREPE features with HuBERT-base 768-dim embeddings @ 50Hz.
Target the IAA gap (0.665 vs 0.740) with pre-trained speech features.

## Procedure
- HuBERT-base-ls960 (Facebook, 95M params), frozen feature extractor.
- Audio resampled 22050 → 16000.
- 768-dim HuBERT + 3 extra (PESTO midi norm, PESTO voicing, CREPE voicing) = 771-dim input.
- BiLSTM head (192 hidden × 2 layers, bidirectional) → onset probability per frame.
- 5-fold CV on Vocadito A1+A2 (40 clips × 2 ann = 80 examples).
- 60 epochs, AdamW lr=1e-3, BCE with positive class weight.

## Results

| fold | F1 |
|---|---|
| 1 | 0.560 |
| 2 | **0.646** |
| 3 | 0.564 |
| 4 | 0.568 |
| 5 | 0.619 |
| **mean** | **0.592** |

Variance is high (0.087 sd). Best fold (2 at 0.646) approaches the cross-annotator
voicing baseline (0.648); other folds are 7-9pp below it.

## Interpretation

HuBERT features don't unlock the data-bound BiLSTM. Probably:
1. **Too few labeled examples** (64 per fold) for a 4M-param BiLSTM head to generalize.
2. **HuBERT is speech-specialized** — fine for vocal humming but the 768-dim features
   capture phonetic structure that's irrelevant for note onsets in humming.
3. **Frozen HuBERT not finetunable** with this little data.

## Comparison

| approach | mean F1 | comment |
|---|---|---|
| voicing+hybrid (heuristic) | **0.648** | cross-annotator mean (best) |
| B50 BiLSTM aug2 (mel+pesto+crepe + ±2 semi) | 0.626 | partial: see B50 report |
| B52 HuBERT BiLSTM | 0.592 | HuBERT doesn't help |
| B42 mel+pesto+crepe BiLSTM | 0.582 | originalBiLSTM result |
| B19 mel-only BiLSTM | 0.562 | first BiLSTM attempt |

The heuristic still wins. Across 4 different BiLSTM variants (B19, B42, B50, B52), none
have beaten the voicing baseline at 0.648. The data is the bottleneck, not the features.

## Next
- **Drop the BiLSTM line entirely** until we have 10× more labeled humming-with-notes data.
- Pursue MERT (music-specific, MERT-v0-public is 95M params trained on music) as a final
  attempt at pre-trained features (B58).
- More likely: focus on segmenter parameter sweep (B57) and offset-detection improvements.

## Status
discard
