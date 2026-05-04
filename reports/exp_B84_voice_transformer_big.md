# exp_B84 — Bigger voice tracker (d=384, 10 layers, 12M params) on same 237 ASAP

## Goal
Test if B76's 94.47% mean accuracy was a model-capacity ceiling or a
data-quantity ceiling by scaling the architecture 6.7× larger.

| variant | params | layers | d_model | best mean acc |
|---|---|---|---|---|
| B76 | 1.78M | 6 | 192 | **0.9447** |
| **B84 (this)** | **12.0M** | **10** | **384** | **0.9438** |

## Procedure
- Same 237 ASAP train pieces, same 4 held-out Romantic
- 80 epochs (vs B76's 50), cosine LR
- AdamW lr=2e-4 weight_decay=5e-4 (extra weight decay for the larger model)
- dropout=0.2 (vs B76's 0.0) to combat the potential overfit risk
- Per-epoch check, save best by val_mean_acc

## Results

| piece | B84 acc | B76 acc | Δ |
|---|---|---|---|
| Beethoven Piano_Sonatas/21-1 | 0.9721 | 0.9739 | -0.18pp |
| Chopin Berceuse op 57 | 0.9480 | 0.9493 | -0.13pp |
| Liszt Sonata | **0.9086** | 0.9078 | **+0.08pp** |
| Schumann Toccata | 0.9464 | 0.9479 | -0.15pp |
| **mean** | **0.9438** | **0.9447** | **-0.09pp** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/sf7f9wq0

## Interpretation
- B84 with 6.7× more parameters basically TIED B76, marginally worse on
  3 pieces and marginally better on Liszt.
- This **confirms the data ceiling** at 237 ASAP train pieces × 1384
  chunks. B76's architecture was correctly sized for this data.
- Further capacity (b=384, 10 layers, 12M params, dropout=0.2) doesn't
  add signal — just more parameters memorising the same training set.

## Decision
**Discard B84** in favor of the smaller B76. No reason to ship the 12M
model when the 1.78M one performs identically.

## Phase E candidate
The path to >94.5% on this task is **more train data** with voice
supervision:
- 4-voice tracks within ASAP (split each hand into sub-voices via HMM)
- MAESTRO MIDI scores rendered + heuristic-voice-split (synthetic supervision)
- Symphonic-MIDI corpora (orchestral scores have explicit voice tracks)

Each of these would need ~1000 additional pieces to start moving the dial.

## Status
discard — confirms B76 is at architectural optimum for the available data.
