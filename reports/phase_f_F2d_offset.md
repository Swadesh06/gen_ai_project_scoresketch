# Phase F-2d — formant offset detector + per-fold Vocadito weights

## Goal

F-2c failure (off20 −0.25 with MIR-ST500-pretrained weights) revealed
that the wrong-domain pretrain destroyed the detector. F-2d retrains
the BiLSTM on Vocadito with per-fold checkpoint saving, then evaluates
each clip with the held-out-fold's weights (proper cross-validation).

## Procedure

1. Modified `scripts/train_formant_offset.py` to accept `save_path`,
   saved per-fold to `checkpoints/formant_offset_vocadito/fold{0..4}.pt`.
2. 5-fold mean F1 in v2 retrain = 0.4900 (slightly above v1's 0.4652;
   random initialisation noise).
3. `scripts/eval_f2d_fold_offset.py`: for each Vocadito clip, find the
   fold that held it out, load that fold's checkpoint, predict offsets,
   score note-level F1.

## Results (40 / 40 clips)

| metric | production baseline | F-2c (MIR-ST500) | F-2d (held-out-fold) | Δ F-2d vs prod |
|---|---|---|---|---|
| noff F1 | 0.6165 | 0.6165 | 0.6165 | 0.0000 |
| off20 F1 | 0.3433 | 0.0937 | **0.1991** | **−0.1442** |

F-2d improves over F-2c by **+0.105** off20 F1 (the right-domain
weights help), but still **−0.144 below the heuristic baseline**.

## Interpretation

The offset-event-level F1 (~0.47 on 5-fold CV) is **fundamentally too
coarse** for the note-level offset20 metric, which requires:
- Onset within ±50 ms of GT (heuristic does this fine)
- Pitch matches GT
- **Note duration within ±20% of GT duration**

The BiLSTM offset predictions are typically within ±5 frames (±50 ms)
of the true offset (that's how event F1 0.47 is achieved). But ±50 ms
on a 0.5-s note is ±10%, which is within tolerance. On a 0.2-s note
it's ±25%, which exceeds the ±20% tolerance and breaks the note match.

The heuristic voicing path benefits from a strong correlation between
voicing dip and true offset — it doesn't predict an absolute offset
time, it inherits the offset from the voicing curve, which is locally
more accurate than a learned BiLSTM's smoothed prediction.

## Decision

**Discard the BiLSTM offset detector for the offset20 production use
case.** Phase F-2e candidates:

1. **Higher temporal resolution**: use 5 ms hop instead of 10 ms in
   the formant features. Halves the ±50 ms event tolerance.
2. **Train a duration-aware loss**: instead of frame-level BCE, optimise
   the F1 metric directly on (onset+duration) pairs.
3. **Use the BiLSTM as a confidence head**, not a replacement: at each
   heuristic-detected offset, the BiLSTM scores whether the offset is
   plausible. Low confidence → extend toward the next voicing dip.
   This combines the two methods rather than swapping.

For now: production keeps the heuristic offset (Phase B+1 baseline).
The BiLSTM artefact is at `checkpoints/formant_offset_vocadito/`.

## Files

- `scripts/train_formant_offset.py` (now saves per-fold checkpoints)
- `scripts/eval_f2d_fold_offset.py`
- `checkpoints/formant_offset_vocadito/fold{0..4}.pt`
- `reports/_phase_f_F2d_offset.json`
