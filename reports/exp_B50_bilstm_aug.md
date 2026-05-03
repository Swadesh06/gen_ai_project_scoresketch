# exp_B50 — BiLSTM with pitch-shift augmentation (5x data)

## Goal
Test if more training data unlocks the data-bound BiLSTM. B42 hit a 5-fold ceiling
at 0.582 with ~80 examples; augment by ±2 semitones (librosa.pitch_shift) to give
5x data per fold = 400 examples total.

## Procedure
- Vocadito 40 clips × 2 annotators × 5 augmentations (-2, -1, 0, +1, +2 semi) = 400 examples.
- 5-fold CV, train on all aug versions of train clips; validate only on the unshifted (steps=0)
  versions of val clips.
- 40 epochs, AdamW lr=1e-3, batch 8 (vs 4 in B42), MelOnsetBiLSTM with mel + PESTO/CREPE feats.

## Results

| fold | F1 |
|---|---|
| 1 | 0.581 |
| 2 | **0.671** |
| 3 | 0.625 |
| 4 | 0.580 |
| 5 | 0.638 |
| **mean** | **0.619** |

## Comparison

| approach | mean F1 | Δ vs voicing 0.648 |
|---|---|---|
| voicing+hybrid (heuristic, current default) | **0.648** | — |
| B50 BiLSTM aug2 (5x data, this exp) | **0.619** | -2.9pp |
| B52 HuBERT BiLSTM (no aug) | 0.592 | -5.6pp |
| B42 mel+pesto+crepe BiLSTM | 0.582 | -6.6pp |
| B19 mel-only BiLSTM | 0.562 | -8.6pp |

## Interpretation

- **Augmentation gives +3.7pp** over B42 (0.582 → 0.619) — the data-bound hypothesis is real.
- **Still 2.9pp below voicing heuristic** — even with 5x data, parametric advantage
  is not enough to beat a tuned heuristic on this domain.
- **Folds vary by 9pp** (0.580 to 0.671) — small validation sets (16 clips) are noisy.

## What's left to push BiLSTM up

The path from 0.619 → 0.65+ needs one of:
1. **More augmentations**: time stretch, room IR, additive noise, formant shift. Could
   give another 5-10x and lift to ~0.64.
2. **Bigger labeled dataset**: combine Vocadito (40 clips) + MedleyDB-Melody (108 clips, has
   F0+voicing but no note onsets — would need pseudo-labeling).
3. **Pre-trained finetuning**: fine-tune HuBERT or MERT instead of frozen feature extraction.
   With 80 training examples this risks catastrophic overfitting.

These are weeks-of-work changes. Given the diminishing returns observed, the cost-benefit
tilts strongly toward keeping the heuristic and putting effort elsewhere.

## Decision
- Discard B50 BiLSTM as a default for Vocadito.
- Continue with `voicing+hybrid` segmenter at F1=0.665 (A1) / 0.630 (A2).
- Pivot to **offset accuracy** improvements (B57) where the gap to IAA ceiling is much larger
  (20.3pp on offset20).

## Status
discard
