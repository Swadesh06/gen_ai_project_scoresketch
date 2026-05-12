# exp_C5 — JSB Chorales LoRA fine-tune of MusicGen-Melody-Large

## Goal

Phase E item 5 per `task_description_v3.md`. B77 validated the LoRA path
on 6 distill pairs but the adapter memorised those 6. C5 trains on real
JSB Chorales (melody-soprano-as-flute, four-voice-arrangement-as-organ)
pairs to test whether the LoRA learns generalisable melody→arrangement
behaviour.

## Procedure

- Data: 371 JSB Chorales rendered via music21 corpus + pyfluidsynth
  (`scripts/prep_jsb_pairs.py`). 315 train / 34 held-out test (90/10
  random split, seed=0).
- Model: MusicGen-Melody-Large (3.3 B base), LoRA r=32 on transformer
  attention modules. 6.29 M trainable params (0.256% of base). fp32 LM
  per the B77 caveat that LoRA needs fp32 attached weights.
- Schedule: 1000 steps, 80 warmup, cosine decay, lr 1e-4, AdamW. Single-
  pair per step (rotates through 4 text prompts).
- Pre-training unit tests (per v3 process discipline): single-batch
  overfit passed (loss 1.47 → 0.0005 in 100 steps). Warmup-vs-total-
  steps assertion (80 < 100) passed.
- Hardware: 14.21 GB peak VRAM (fits 16 GB card). Wall = 44 min.

## Results

| metric | value |
|---|---|
| train_loss_min | 1.008 |
| train_loss_final | 1.454 |
| test_loss_mean (held-out) | **1.388** |
| vram_peak_gb | 14.21 |
| wall_train_s | 2636 (44 min) |

Loss curve analysis (`scripts/analyze_jsb_loss.py`):
- Per-prompt minima: 1.07, 1.11, 1.20, 1.09 — model can fit individual
  prompts to ~1.1.
- Per-prompt medians: 1.43, 1.41, 1.42, 1.45 — across pairs, fits
  stagnate at ~1.4.
- First-quartile mean = 1.379; last-quartile mean = 1.390 (delta +0.012)
  → **plateau**.
- Test loss = 1.388 ≈ train mean → no overfitting, no generalisation
  gap to close. The model is **capacity-limited** under r=32.

## Comparison to B77

B77 (synthetic distill, r=32, 300 steps): final loss 0.73 with full
memorisation of 6 pairs. C5 (real pairs, r=32, 1000 steps): test loss
1.39. The numbers aren't apples-to-apples — B77 measures memorisation
of its training set; C5 measures held-out generalisation. C5 is more
honest.

## Interpretation

The plateau pattern across all 4 prompts at ~1.4 (while individual pair
fits reach 1.0) means the LoRA capacity is enough for individual pairs
but not enough to model the variation across 315 chorales. The LoRA
isn't learning a "generic Bach-chorale arrangement" function — it's
learning to fit each pair locally.

For a productive next iteration:
- **r=64 LoRA** to double trainable capacity (~12 M params, would fit
  in ~14 GB peak).
- **Or larger dataset**: JSB chorales are pretty uniform (4-voice,
  similar tessitura, all Bach). Mixing in Lakh MIDI or arrangement
  pairs from a more varied corpus may push generalisation.
- **Or longer melody window**: target_dur_s=10 may be too short for
  MusicGen to learn structural patterns. 30 s windows + chunking might
  help.

## Decision

**Keep behind the lora_adapter flag**, don't promote as default. B77's
distill-trained adapter remains the demo-mode default. C5 is the more
honest version of the same pipeline (real pairs, no distill) but its
generalisation isn't strong enough to be the new headline.

Phase F-4 candidates (in priority order):
1. C5b with r=64, same data (test whether capacity is the constraint)
2. C5c with r=32 on Lakh MIDI subset (test whether data variance helps)
3. C5d with r=64 + Lakh + longer windows

## Files

- `scripts/exp_C5_jsb_lora.py`
- `scripts/prep_jsb_pairs.py`
- `scripts/analyze_jsb_loss.py`
- `reports/_exp_C5_jsb_lora.json`
- `checkpoints/musicgen_lora_c5_jsb/step_1000/` (rolling, last 4 kept)
- WandB: humscribe-v3.2/runs/<id>
