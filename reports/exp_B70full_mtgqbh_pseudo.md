# exp_B70full — Aggressive B70 (hidden=192, 40 epochs) on Vocadito + MTG-QBH pseudo

## Goal
B70 v2 used reduced config (hidden=96, 15 epochs) to fit within a single
session. B70 full uses the original aggressive config (hidden=192, 40 epochs)
to test if more capacity + more training closes the gap to the heuristic.

## Procedure
- Same as B70 v2 in `scripts/exp_B70full_mtgqbh_pseudo.py`
- 158-clip combined train (40 Vocadito real + 118 MTG-QBH pseudo)
- 5-fold CV on Vocadito only, two heads per fold (combined + voconly control)
- Hidden=192 (was 96), n_epochs=40 (was 15)

## Results

| fold | combined val F1 | voconly val F1 | Δ |
|---|---|---|---|
| 0 | 0.4402 | 0.4134 | +2.68pp |
| 1 | 0.3640 | 0.3064 | +5.76pp |
| 2 | 0.2597 | 0.3513 | **-9.16pp** |
| 3 | 0.3481 | 0.2957 | +5.24pp |
| 4 | 0.3998 | 0.3928 | +0.70pp |
| **mean** | **0.3624** | **0.3519** | **+1.05pp** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/zl5u895s

## Vs B70 v2 (smaller config)

| variant | hidden | epochs | combined | voconly | pseudo gain |
|---|---|---|---|---|---|
| B70 v2 | 96 | 15 | 0.3715 | 0.3472 | +2.43pp |
| **B70 full** | **192** | **40** | **0.3624** | **0.3519** | **+1.05pp** |

The fuller config is **slightly worse** in both absolute and pseudo-gain
terms. The 192/40 setup overfits faster on the augmented 158-clip train set,
which has more high-variance noise from the 118 pseudo labels.

## Interpretation
- **Pseudo-label gain is real but modest** (+1.05pp mean across 5 folds)
- **Fold 2 regression** (-9.16pp) is striking — the pseudo labels actively
  hurt that specific Vocadito subset. Pseudo labels have piece-dependent
  quality that the fixed BiLSTM can't reason about.
- **Absolute F1 is ~30pp below the heuristic** (0.36 vs 0.665). All
  learned approaches on 40-clip Vocadito hit this same wall.

## Decision
**Discard.** Item 5's pass criterion (≥ 0.69) is missed by ~33pp. B70
full doesn't change the conclusion from B70 v2. Phase D's 4 BiLSTM
attempts (B19, B42, B50, B70 v2, B70 full, B72 in flight) consistently
plateau at 0.35-0.55 F1 on this dataset.

## Phase E real fix
Real-label data scale-up is the only path:
- Hand-aligned 5-10 MTG-QBH clips in MuseScore (~30 min/clip × 10 ≈ 5 h)
  → real labels for those clips → +148 clip train set with mixed real/pseudo
- MedleyDB-Melody if registration cleared
- Synthesize humming via TTS conditioned on melody (longer Phase F shot)

## Status
discard — 4th confirmation of the 40-clip Vocadito data ceiling for learned
voicing. Keep B70 v2 baseline (smaller, faster, marginally better).
