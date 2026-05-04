# exp_B72 — BiLSTM + 4× MIDI augmentation (Vocadito + MTG-QBH pseudo)

## Goal
Test if heavy on-line augmentation (4× per-clip per epoch with pitch
shift, time stretch, SpecAug, noise) closes the gap between B70's BiLSTM
(combined 0.371 / voconly 0.347) and the heuristic 0.665.

Same train set as B70 (40 Vocadito real + 118 MTG-QBH pseudo, 158 clips
total). Same 5-fold CV split. Architecture: BiLSTM 3-layer, hidden=256,
dropout=0.3.

## Procedure
- 80 epochs per fold, 5 folds = 400 epoch passes
- Each epoch processes ~5 samples per clip (1 original + 4 augmented)
  through the full BiLSTM forward+backward
- AdamW lr=1e-3 cosine, weight_decay=1e-4

## Results (PARTIAL — killed at fold 0 ep 51 of 80, projected ETA was 22h)

| fold | val F1 |
|---|---|
| 0 (best so far) | **0.4709** |
| 1-4 | not run |

Train loss at ep 51: 0.198. Best val F1: 0.471 — **higher than B70 v2's
fold-0 baseline of 0.44**, suggesting augmentation does help on this fold.

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/5a7tq2k9

## Why killed
At observed 224 s/epoch (vs estimated 60 s/epoch when launched), the full
5-fold run would have taken 22 hours. The partial fold-0 result already
suggests augmentation helps, but a single-fold sample is too noisy to act
on, and the absolute 0.471 is still 19pp below the heuristic 0.665.

The decision to kill was driven by:
1. The full result wouldn't change the Phase D conclusion (heuristic >
   any learned approach on 40-clip Vocadito).
2. 22h of GPU time better spent on Phase E experiments with real-label
   data.

## Phase E next step (if pursued)
1. Reduce per-epoch augmentation from 4× to 2× → 11h ETA
2. Reduce epochs from 80 to 30 → fits in ~5h
3. Or keep settings but skip pseudo labels (Vocadito-only) → 4h ETA
4. Best path: hand-align 5-10 MTG-QBH clips → 158-clip mixed real/pseudo
   train set → re-run the smaller variant

## Status
killed — partial fold-0 result (0.471) suggests augmentation helps but the
22h ETA was excessive. Phase E should re-run with fewer epochs / fewer
augmentations once we have real-label data.
