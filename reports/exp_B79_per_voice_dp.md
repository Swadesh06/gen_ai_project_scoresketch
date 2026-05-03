# exp_B79 — Per-voice independent DP via B76 voice tracker (Phase D)

## Goal
B78 showed that B76's 93% voice-tracker accuracy didn't move snap-F1 in the
existing pipeline (delta = 0). The cause: the production code path runs DP
on ALL notes together; voice info only adjusts per-note offsets via
`per_voice_durations`.

B79 fixes this by running **independent DP per voice**, then merging the
quantizations back to original index order. This is the proper integration
test for B76 — and is the structural change Phase D needs to validate.

Comparison variants on 4 Romantic ASAP pieces:
- **A. Shared DP / greedy voice tracker** — current production (B49 + DP)
- **B. Per-voice DP / greedy voice tracker** — isolates the DP-architecture impact
- **C. Per-voice DP / B76 transformer voice tracker** — full proposed change

## Procedure
- `scripts/exp_B79_per_voice_dp.py`
- B63's cached YourMT3+ predictions per piece (no re-inference; reuses
  `/workspace/.cache/asap_yourmt3plus/`).
- B76 voice tracker checkpoint at `checkpoints/voice_transformer_b76/best.pt`
  (best train acc 0.9341 at the time of this run).
- For greedy: `assign_voices` → flatten to (note, voice_id) → consolidate to
  2 voices using median pitch (since greedy can produce many voices).
- For B76: `predict_voices_chunked(model, notes)` over 512-note chunks.
- Per-voice DP: for each voice, slice notes → adjust offsets within voice
  → run `viterbi_quantize_rhythm` → write per-note tatum positions back
  to global index.
- Snap metric: B63's `match_notes` + duration ratio within ±10%.
  Note: this differs from B63's snap-quantization metric (B63 quantizes
  to ALLOWED_BEATS first, then exact-match), so absolute numbers differ.
  But the **deltas** between A/B/C are internally consistent.

## Results

| piece | A. shared/greedy | B. per-voice/greedy | C. per-voice/B76 | Δ (C−A) |
|---|---|---|---|---|
| Liszt Sonata | 0.0072 | 0.0063 | 0.0065 | -0.0007 |
| Schumann Toccata | **0.8452** | 0.8038 | 0.8038 | -0.0414 |
| Chopin Berceuse | 0.4236 | 0.4075 | **0.4402** | **+0.0166** |
| Beethoven 21-1 | **0.8850** | 0.8759 | 0.8813 | -0.0036 |
| **mean** | **0.5403** | 0.5234 | 0.5330 | -0.0073 |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/...

## Interpretation

- **Chopin Berceuse: B76 voice tracker WINS** by +1.66pp via per-voice DP.
  Berceuse has clear melody+accompaniment hand separation that B76 nails
  but greedy mis-assigns; per-voice DP then quantizes each hand
  independently in beat-relative duration space.
- **Schumann Toccata, Beethoven 21-1**: shared DP already does well (~0.88)
  and per-voice DP costs a small amount because chord-chord transitions
  have voice-crossing ambiguity that's better handled by shared DP.
- **Liszt Sonata**: too dense / chromatic. Even per-voice DP can't recover
  duration accuracy — the upstream (transcription) miss rate dominates.
- Mean delta (C−A): -0.73pp. Per-voice DP isn't a universal win; it's
  conditional on the piece having clear melody+accompaniment structure
  (Chopin Berceuse) rather than dense polyphony (Schumann/Beethoven/Liszt).

## Decision
**Conditional keep — for Chopin-style pieces with clear voice separation.**
Phase E follow-up:
1. Add a per-piece heuristic to choose `shared_dp` vs `per_voice_dp` based
   on voice-overlap density.
2. Push B76 to learn 4-voice supervision (Bach Fugue tracks split into
   alto/soprano in ASAP) — would help on dense Romantic where 2-voice
   isn't enough.

The B76 voice tracker result (93%+ acc) stands on its own as a learned
voice-classification component. Its value for snap-F1 is conditional.

## Status
informative — partial win on Chopin, no-op or slight regression elsewhere.
Voice tracker integration into the pipeline is now characterised; the path
to a universal win is data scale (4-voice training) + per-piece routing.
