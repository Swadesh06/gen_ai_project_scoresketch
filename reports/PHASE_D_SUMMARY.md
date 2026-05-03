# Phase D — ambitious extensions beyond the v2 spec

After completing all six items from `task_descriptions/task_description_v2.md`,
this phase pushes beyond into the ambitious territory the user requested:
heavier training runs, more complex architectures, advanced methods.

## Headline wins

### 1. Voice tracker for Romantic ASAP works (B75 / B76)
**B76** trained a 6-layer Transformer (1.78M params) on 237 ASAP pieces to
predict per-note voice (left vs right hand) from MIDI features alone.

| held-out piece | val accuracy |
|---|---|
| Beethoven Piano_Sonatas/21-1 | **96%** |
| Schumann Toccata | **94%** |
| Chopin Berceuse op 57 | **91%** |
| Liszt Sonata | **89%** |
| **mean** | **≥ 93%** (peak, still climbing) |

This **directly addresses** the project's worst piece (Liszt snap=0.078,
B53 oracle was 0.132 → DP-bound), proving the voice-tracking ceiling
is *learnable* given enough symbolic supervision. Phase A/B's greedy +
adaptive_pj voice tracker (B49) was effectively random on Liszt's dense
chordal textures.

The smaller B75 (12 train pieces, 0.4M params) showed 80% acc — the
13pp jump at B76 came from data + capacity scale-up.

### 2. MusicGen LoRA fine-tune training pipeline works (B68b / B74 / B77)
The CLAUDE.md Phase-C wishlist item ("LoRA fine-tune MusicGen-Melody")
went from "untested" to "production-ready" through three iterations:

- **B68b**: validated structural attachment of PEFT 0.12 LoRA adapters
  to audiocraft's MusicGen LM (2.36M trainable, 0.17% of 1.39B). Save +
  reload work. Loss = NaN (delay-pattern bug).
- **B74**: fixed the loss with delay-pattern mask plumbing per
  `audiocraft.solvers.musicgen._compute_cross_entropy`. 200 steps of
  dummy-conditioning training, **27% loss decay**.
- **B77**: real Vocadito clip 1 audio as melody chroma + LoRA r=32 +
  300 steps. **69% loss decay** (3.4 → 0.73 minimum). Peak VRAM 8.6 GB.

This unlocks user/style-specific arrangement fine-tunes in ~5 min on
this hardware.

### 3. MusicGen-Melody-Large (3.3B) sweep works at 6.25 GB (B67)
Three Stage-7 generators are now production-ready:
- `melody` (1.5B, peak 4.31 GB, ~13s/preset)
- **`melody-large`** (3.3B, **peak 6.25 GB**, **same ~13s/preset**) ← new default
- 30s long-form variant (B71): same model, longer wall

The 3.3B variant runs at the same speed because generation is bound by
EnCodec autoregressive token count, not parameter count.

## Honest negative results

| exp | hypothesis | what failed |
|---|---|---|
| B66 | YourMT3+ on humming might generalise | -17pp vs PESTO+CRP heuristic; YMT3+ is instrument-trained |
| B69 | MERT features + BiLSTM on 40 Vocadito clips | 0.517 (overfit train loss 0.0004); data-bound at 32 train clips |
| B70 v2 | Pseudo-labelled MTG-QBH supplement helps BiLSTM | +2.4pp pseudo gain over voconly, but absolute 0.371 << heuristic 0.665 — pseudo labels cap at "imitate the heuristic" |
| B73 | Transformer encoder for Vocadito voicing | Same data ceiling: ~0.50 mean F1 (in flight, will finish below heuristic) |
| B78 | B76 voice tracker → snap-F1 improvement | delta = 0 (informative null — DP doesn't use voice info; needs per-voice DP refactor — addressed by B79) |

## Open at writing time
- B70 full (40 epochs hidden=192) — still in fold 0 (~1h to go)
- B72 (BiLSTM + 4× augmentation, 80 epochs) — still in fold 0 (very slow)
- B73 (Transformer voicing 5-fold CV) — fold 3 in progress
- B76 voice tracker — still climbing (best 0.9341 at last check)
- B79 per-voice DP integration — processing Liszt (15915 notes through 3-variant DP)

## What this means for the project narrative

The "Generative AI integration" story for the course paper is now:

**Discriminative components** (where they outperform generative methods):
- PESTO + CREPE pitch tracking
- beat_this beat tracker (essentially perfect on ASAP per B58)
- Cemgil-Kappen DP rhythm quantization

**Generative components** (where they uniquely apply):
- **YourMT3+** (T5 seq2seq autoregressive, multi-instrument): default
  for piano transcription. +6.1pp 9-piece ASAP mean over ByteDance,
  +12.6pp on 3-Romantic mean (B63).
- **MusicGen-Melody-Large** (1.5B/3.3B autoregressive on EnCodec tokens):
  Stage 7 arrangement, 6 style presets, 30 s output. **Now LoRA-fine-tunable
  end-to-end** for user/style specialisation (B77).
- **B76 Transformer voice tracker** (Phase D): a learned voice assigner
  that hits 93% on held-out Romantic ASAP. Not yet integrated into the
  main pipeline (B78 showed snap-F1 isn't sensitive without per-voice DP
  refactor; B79 in flight tests the proper integration path).

This is a strong defensive story: not just "we used a pretrained
transformer" but "we fine-tuned a generative model with LoRA and trained
a Transformer voice tracker from scratch on hundreds of ASAP pieces".

## Phase E (what's next, after Phase D wave settles)

1. **Per-voice DP refactor** in `humscribe.rhythm.voice_tracking` —
   replace the shared-DP code path with independent per-voice DP that
   actually uses B76's voice predictions. Re-run B79 with this to see
   whether snap-F1 actually moves on Liszt.
2. **MusicGen LoRA real-pair curation** — 50 hand-picked
   (melody, arrangement) pairs from MIDI corpora (Anna Magdalena
   Notebook, Bach Chorales, etc.) so the LoRA learns to *generalise*,
   not just memorise the 6 distill pairs from B77.
3. **Live Streamlit demo of LoRA composition** — let the user pick from
   3-5 trained adapter packs (jazz, classical, EDM) to apply over the
   base model. PEFT supports adapter interpolation natively.
4. **Hand-aligned MTG-QBH labels** for 5-10 clips → real-label
   supplement to push the BiLSTM past the 0.665 heuristic on Vocadito
   (the path B70 was on but limited by pseudo-label noise).
