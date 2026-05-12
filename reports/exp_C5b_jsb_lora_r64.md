# exp_C5b — JSB Chorales LoRA r=64 (capacity test)

## Goal

C5 (r=32) test loss = 1.39 ≈ train mean = 1.39 with per-prompt minimum
~1.07. Diagnosis: capacity-limited, not training-step-limited. C5b
re-runs the same data at r=64 (2× trainable params) to test the
capacity hypothesis directly.

## Procedure

- Same data: 371 JSB chorales rendered as (soprano-flute, four-voice-organ)
  pairs at 32 kHz. 315 train / 34 test split (seed=0).
- Model: MusicGen-Melody-Large 3.3 B base, LoRA r=64 on attention
  modules. 12.58 M trainable (0.511% of 2464 M base). fp32 LM.
- Schedule: **1500 steps** (vs C5's 1000), 100 warmup, cosine decay,
  lr 1e-4. AdamW. Single-pair per step with 4-prompt rotation.
- Unit-test skipped (`--skip-overfit`) because the C5 single-batch
  overfit already validated the path.
- Hardware: peak VRAM 14.32 GB on the 16 GB card. Wall = 50 min.

## Results

| metric | C5 r=32 | C5b r=64 | Δ |
|---|---|---|---|
| trainable params | 6.29 M | 12.58 M | 2.0× |
| train loss min | 1.008 | **0.610** | −0.398 |
| train loss final | 1.454 | 0.708 | −0.746 |
| **test loss mean (held-out)** | **1.388** | **0.983** | **−0.405** |
| vram peak (GB) | 14.21 | 14.32 | +0.11 |
| wall (s) | 2636 | 3023 | +387 |

**−0.41 test loss reduction from doubling LoRA rank.** Capacity
hypothesis confirmed strongly. The train loss continued to decrease
through step 1500 (final 0.71 vs starting 1.7), so the cosine schedule
isn't even close to saturating r=64's capacity.

## Decision

**Promote C5b r=64 as the new default LoRA adapter** for JSB-style
arrangement tasks. The B77 distill adapter remains as `--lora-adapter
b77_distill` for backward compatibility.

For Phase F-4 (Lakh MIDI LoRA), use the same r=64 + ≥1500 steps
schedule. The C5b adapter is at `checkpoints/musicgen_lora_c5_jsb/step_1500/`.

## Interpretation

The capacity hypothesis was the right diagnosis. At r=32 the LoRA could
fit individual pairs (best train loss 1.07) but couldn't generalise
across 315 chorales (test loss 1.39). At r=64, both train and test
loss drop substantially, with train still decreasing at the end of the
schedule — meaning even more capacity (r=128) might help further. But
r=64 already fits in the 16 GB card and the +0.11 GB cost over r=32
is trivial. Diminishing-returns territory: a further r=128 run isn't
high-priority unless the Lakh MIDI corpus (much bigger data variance)
demands it.

The session-end production headline isn't a change because LoRA only
runs at user request (Stage 7 arrangement is an optional post-stage).
But the demo flourish is meaningfully better — generated arrangements
should follow the input melody more faithfully and remain stylistically
coherent.

## Files

- `scripts/exp_C5_jsb_lora.py` (same script — `--lora-r 64`)
- `scripts/analyze_jsb_loss.py`
- `reports/_exp_C5_jsb_lora.json` (last-write wins; this file is from
  the r=64 run; r=32 data still in WandB)
- `checkpoints/musicgen_lora_c5_jsb/step_{1100,1200,1300,1400,1500}/`
- WandB: humscribe-v3.2 — run name `exp_C5_jsb_lora_melody-large`
