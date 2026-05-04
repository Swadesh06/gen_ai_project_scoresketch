# exp_B83 — B76 voice tracker + heavy MIDI augmentation (Phase D)

## Goal
Test if heavy on-line augmentation (pitch shift ±5 semitones, time stretch
0.85-1.15×, voice-swap p=0.5, note dropout p=0.1, onset jitter ±15ms,
2× per-chunk) can push past B76's 94.47% mean acc on Romantic ASAP.

Same train/val split as B76 (237 ASAP train + 4 Romantic held-out).
Same architecture (6-layer Transformer, d=192). Only difference: every
training chunk passes through the augmentor before backprop.

## Results

| metric | B83 (this) | B76 baseline | Δ |
|---|---|---|---|
| Best mean val acc | **0.9448** | 0.9447 | **+0.01pp** (essentially tied) |
| Beethoven 21-1 | 0.9706 | 0.9739 | -0.33pp |
| Chopin Berceuse | 0.9449 | 0.9493 | -0.44pp |
| Liszt Sonata | 0.9032 | 0.9078 | -0.46pp |
| Schumann Toccata | **0.9603** | 0.9479 | **+1.24pp** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/9pyc61x7

## Interpretation
- B83 essentially **ties** B76 on mean — augmentation gives +1.24pp on
  Schumann Toccata but takes 0.3-0.5pp from each of the other three.
- Pitch shift ±5 semitones likely confuses Liszt (a deeply chromatic
  piece) more than it helps. Schumann (a faster, more rhythmic piece)
  benefits from the time-stretch / dropout regularisation.
- Combined with B84 (12M params, 0.9438 — also ties B76), the takeaway
  is clear: **B76's 94.47% is at the data-quantity ceiling, not the
  capacity or augmentation ceiling**.

## Decision
**Discard** B83 in favor of the simpler B76. The piece-level trade
(better Schumann, slightly worse on the other 3) doesn't beat B76's
balanced profile. Liszt was the worst-performing piece on which we'd
hope to see augmentation help — it didn't.

## Phase D wave-end conclusion
Three voice tracker variants now characterised:
- B76 (baseline, 1.78M params, 50 epochs): **0.9447** (production)
- B83 (B76 + heavy aug, 60 epochs): 0.9448 (tie, discarded)
- B84 (12M params, 80 epochs): 0.9438 (worse, discarded)

The data ceiling is real. Phase E path to >94.5% is more train data
with voice supervision (4-voice splits, MAESTRO, Symphonic-MIDI), not
better/bigger model.

## Status
discard — ties B76 on mean; loses on 3 of 4 held-out pieces.
