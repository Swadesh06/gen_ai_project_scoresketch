# Phase F-2c — formant offset detector wired into Vocadito gate (negative)

## Goal

Phase F-2c production-integration step. Replace the voicing-based
offset estimator with the F-2 formant BiLSTM in the Vocadito pipeline
and verify the offset20-F1 lift (heuristic baseline 0.439 → IAA ceiling
0.642 = 22pp gap).

## Procedure

`scripts/eval_formant_voc_offset.py`:
- Load the `checkpoints/formant_offset_mirst500.pt` checkpoint
  (MIR-ST500 pretrained, test F1 0.30 — the F-2b artifact).
- For each Vocadito clip: run the production PESTO+CREPE-voicing path
  to get onsets; run the formant BiLSTM on cached formant features to
  predict offsets; pair them.
- Score note-level F1 at ±50 ms onset tolerance, with and without ±20%
  duration offset constraint.

## Results

| metric | production baseline | F-2c (MIR-ST500 weights) | Δ |
|---|---|---|---|
| noff F1 | 0.6165 | 0.6165 | 0.0000 |
| **offset20 F1** | **0.3433** | **0.0937** | **−0.2496** |

40/40 clips evaluated. **Discard at these weights.** The formant
detector with MIR-ST500-pretrain produces offset predictions that are
worse than the heuristic on every single clip.

## Why this failed

The checkpoint used is `formant_offset_mirst500.pt` (test F1 0.30 on
MIR-ST500 itself, per F-2b report). It was never fine-tuned on Vocadito.
The MIR-ST500 corpus is pop songs with full instrumental backing,
where vocal formants are masked by drums/bass. Vocadito is solo voice.
A model that learns to localise offsets in MIR-ST500's polyphonic
mixture has no reason to generalise to Vocadito's clean vocal.

The fix is obvious in hindsight: **use the F-2 base Vocadito-trained
weights** (5-fold mean F1 0.4652). But the F-2 trainer (`scripts/train_formant_offset.py`)
doesn't save per-fold checkpoints — it discards the model after
computing val F1.

## Decision

Discard the MIR-ST500-pretrained checkpoint as a Vocadito offset
source. Phase F-2d:

1. Modify `scripts/train_formant_offset.py` to save per-fold
   checkpoints to `checkpoints/formant_offset_vocadito_fold{0..4}.pt`.
2. Re-run the 5-fold CV training (saves per-fold weights).
3. Re-run F-2c using the held-out-fold's checkpoint to predict offsets
   for the clips in that fold. Average across folds for the headline
   number.
4. If the F2c offset20 F1 is then ≥ 0.45 (= +0.10pp over heuristic
   0.343), promote as production default.

Estimated effort: ~30 min (training is fast). Deferred to next session.

## Files

- `scripts/eval_formant_voc_offset.py`
- `checkpoints/formant_offset_mirst500.pt` (kept for ablation, but not
  used in production)
- `reports/_phase_f_F2c_offset.json` (40-clip per-clip data)
