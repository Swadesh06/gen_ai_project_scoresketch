# exp_B69 — MERT-features BiLSTM voicing on Vocadito (Phase C)

## Goal
Replace HuBERT (B52: F1=0.592) with **MERT-v1-95M** (Music-trained BERT,
Yizhi Li et al. ICLR 2024, [arXiv:2306.00107](https://arxiv.org/abs/2306.00107))
as the encoder for a BiLSTM voicing/onset detector. MERT is pretrained
specifically on music with two heads (acoustic + harmonic), so its
embeddings should capture vocal note structure better than speech-trained
HuBERT.

Pass criterion: Vocadito A1 noff F1 (5-fold CV) ≥ 0.69 (current heuristic
0.665, IAA ceiling 0.740).

## Procedure
1. Per-clip cached features: MERT-v1-95M last-4-layer mean (~768D) plus
   PESTO/CREPE pitch+voicing resampled to MERT's 75Hz timebase. Total
   feature dim: 772.
2. Frame-level binary "is in note" labels from Vocadito A1 annotations.
3. BiLSTM 2-layer, hidden=192, BCE loss, AdamW lr=1e-3 weight_decay=1e-4,
   30 epochs, no augmentation.
4. 5-fold CV over the 40 Vocadito clips (32 train, 8 val per fold).
5. Inference: sigmoid → threshold 0.5 → exit-side hysteresis intervals →
   median PESTO pitch per interval → mir_eval no-offset F1.

## Results

| fold | val F1 |
|---|---|
| 0 | 0.5999 |
| 1 | 0.4752 |
| 2 | 0.5008 |
| 3 | 0.4969 |
| 4 | 0.5143 |
| **mean ± std** | **0.5174 ± 0.0431** |

Train loss converged to 0.0004 by epoch 28 (severe overfitting on the
32-clip train splits — model memorized the training data).

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/o8j7t0k5

| approach | Vocadito A1 noff F1 |
|---|---|
| Heuristic PESTO+CRP voicing (current default) | **0.665** |
| B52 HuBERT BiLSTM | 0.592 |
| **B69 MERT BiLSTM (this)** | **0.517** |
| B69 fold 0 best | 0.600 |
| IAA ceiling | 0.740 |

## Interpretation
- MERT-95M does NOT generalise from 32 train clips to 8 val clips for
  this onset/voicing task. Mean F1 0.517 is 15pp below the heuristic.
- Training loss collapsing to 0.0004 confirms severe overfitting — the
  model memorises 32 clips' frame-level voicing patterns and fails on
  unseen clips.
- B52 HuBERT got 0.592 — slightly better than B69 here. Both are well
  below the heuristic.
- The gap is fundamentally a *data quantity* problem. 40 Vocadito clips
  is too few to train any 1M+ parameter learned model from scratch on
  this binary segmentation task.

## Decision
**Discard.** MERT features alone don't beat the heuristic — same conclusion
as B52 HuBERT. The next experiment (B70) tests whether **more data** via
MTG-QBH pseudo-labels closes the gap.

## Next
B70: pseudo-label 118 MTG-QBH humming clips and combine with the 40
Vocadito clips for a 158-clip train set. The bet is that more humming
data of the right distribution (not better features) is what's needed.

## Status
discard
