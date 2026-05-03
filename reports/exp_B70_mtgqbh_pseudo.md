# exp_B70 — MTG-QBH pseudo-label + BiLSTM training (item 5 substitute)

## Goal
Per `task_description_v2.md` §Work item 5, which originally targeted
MedleyDB-Melody (registration-gated). MTG-QBH is the natural substitute
in the same domain (humming, monophonic vocal, Zenodo-public, no auth
needed). Same recipe as the spec:
1. Pseudo-label MTG-QBH onsets via the existing voicing+pitch heuristic.
2. Combine with Vocadito (real labels) → enlarged train set.
3. Re-train mel-BiLSTM, compare to heuristic baseline.

Pass criterion: Vocadito A1 noff F1 ≥ 0.69 (current heuristic 0.665).

## Procedure

### Bootstrap MTG-QBH
- `humscribe.datasets.mtg_qbh.MTGQBH` Zenodo-direct downloader. 118 humming
  clips bootstrapped to `~/datasets/mtg_qbh/audio/`.

### Feature extraction
- Same recipe for both real-label Vocadito and pseudo-label MTG-QBH:
  mel (64 bins, hop 10ms, 22.05 kHz) + PESTO + CREPE on a common 100Hz
  timebase. Feat dim = 68. Cached to `/workspace/.cache/voc_qbh_features/`.

### Pseudo-label generation
- For each MTG-QBH clip, run the same `humscribe.pitch.ensemble.track_pitch_hybrid_voicing`
  + `segment_pitch_to_notes` pipeline that's currently in production at
  Vocadito A1 F1=0.665. The resulting (intervals, pitches) become the
  pseudo labels.

### Training
- 5-fold CV on Vocadito A1 only (the validation gold standard).
- For each fold, two heads:
  - `combined`: train on Vocadito_train (32) + MTG-QBH (118) = 150 clips
  - `voconly`: train on Vocadito_train (32) only — control
- BiLSTM 2-layer, hidden=96 (B70 v2 reduced from initial hidden=192 for
  speed; full version follow-up B70_full ran with hidden=192).
- BCE per-frame loss, AdamW lr=1e-3 weight_decay=1e-4, 15 epochs.
- Inference: sigmoid → 0.5 threshold → exit-side hysteresis intervals →
  median PESTO pitch per interval → mir_eval no-offset F1.

## Results (B70 v2: hidden=96, 15 epochs)

| fold | combined val F1 | voconly val F1 | delta |
|---|---|---|---|
| 0 | 0.4343 | 0.4119 | +0.022 |
| 1 | 0.3189 | 0.2991 | +0.020 |
| 2 | 0.3332 | 0.3349 | -0.002 |
| 3 | 0.3880 | 0.3062 | +0.082 |
| 4 | 0.3833 | 0.3838 | -0.001 |
| **mean** | **0.3715** | **0.3472** | **+0.024** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/gcghltt9

## Vs criteria
- Pseudo-label gain: **+2.4pp** (combined > voconly mean across folds).
  Real signal that MTG-QBH pseudo-labels HELP, not regress.
- Absolute combined F1: **0.3715**, well below heuristic 0.665 (-30pp)
  and IAA 0.740 (-37pp). Fails item-5 pass criterion of 0.69.
- Train losses converged early (epoch 3-4) → BiLSTM is data-bound.

## Interpretation
- The pseudo-label idea is *directionally correct* — combining 158 train
  clips outperforms 32 by 2.4pp, even when the extra 118 are noisy
  pseudo labels from the heuristic.
- But pseudo-labels have a **fundamental ceiling**: they teach the BiLSTM
  to *imitate the heuristic*, not exceed it. This is why combined F1
  (0.37) is far below heuristic F1 (0.67) — the BiLSTM hasn't learned
  to do anything the heuristic doesn't already do. Plus it's smaller
  capacity than the heuristic's full PESTO+CREPE+voicing pipeline.
- For this approach to beat the heuristic, the train set needs **real
  labels** at scale (e.g. hand-aligning 50 MTG-QBH clips in MuseScore,
  or sourcing MedleyDB-Melody via registration).

## Decision
**Discard** for production. Item 5 pass criterion (≥ 0.69) not met by
~30pp. The pseudo-label approach validates as a small positive boost
(+2.4pp combined > voconly) but doesn't close the gap to the heuristic.

The follow-up experiment B70_full (40 epochs, hidden=192) and B72
(B70 + augmentation) are running to test whether more capacity / more
augmentation closes the absolute gap. Even if it does, the underlying
ceiling remains: pseudo-labels cap at "imitate the heuristic".

## Status
discard for production; informative for Phase D recommendation
(real-label dataset scale-up).
