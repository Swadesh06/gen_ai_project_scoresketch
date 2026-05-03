# exp_B85 — Learned offset corrector for Vocadito (Phase D)

## Goal
Target the unfixed Vocadito offset20 gap: heuristic F1 = 0.439 vs IAA = 0.642.
Train an MLP that takes per-note features (predicted duration, position,
voicing trace ±15 frames around predicted offset) and predicts the
correction (gt_offset − pred_offset).

## Procedure
- 5-fold CV on 40 Vocadito clips
- Pair each predicted note with its mir_eval-matched GT note (~ 1500 pairs total)
- 32-dim input: log(dur), position, 30-frame voicing window
- 3-layer MLP, hidden=128, dropout=0.2, AdamW lr=1e-3
- 100 epochs per fold

## Results
| metric | value |
|---|---|
| baseline heuristic offset20 F1 (5-fold CV) | 0.4394 |
| **B85 MLP-corrected offset20 F1** | **0.4328** |
| delta | **−0.66pp** |
| vs IAA ceiling 0.642 | gap +0.21 |

WandB: in-flight run id (search exp_B85_offset_corrector).

## Interpretation
Slight regression. The heuristic is hard to beat at this data scale
(~1500 matched note pairs across all 40 clips). The model learned
something but it generalises worse than the simple voicing-end rule.

This confirms the pattern from B69/B70/B72/B73:
- BiLSTM voicing on Vocadito features: 0.51-0.53 (vs heuristic 0.665)
- Transformer voicing on Vocadito: 0.50
- BiLSTM with MTG-QBH pseudo-labels: 0.37
- BiLSTM with augmentation: in flight, similar pattern expected
- **MLP offset corrector: 0.43 (vs heuristic 0.44)**

All learned approaches on the 40-clip Vocadito training set hit a data
ceiling around or below the heuristic baseline.

## Decision
**Discard.** The fix path for offset20 isn't a learned model on existing
Vocadito; it's either:
1. Real-label data scale-up (hand-aligned MTG-QBH, MedleyDB)
2. A different kind of model (e.g. self-supervised audio features +
   downstream finetuning at scale)
3. Accept the IAA ceiling — humming offset prediction has a fundamental
   ambiguity that even humans disagree on (B51 IAA offset20 = 0.642).

## Status
discard — yet another data-bound learned approach failure on Vocadito.
Phase E only-fix is real-label dataset scale-up.
