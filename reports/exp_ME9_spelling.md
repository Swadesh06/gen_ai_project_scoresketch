# ME-9 — line-of-fifths enharmonic spelling — discard at this tuning

## Goal

Phase E item 7 ME-9 per `task_description_v3.md`. Implement Temperley
2001's line-of-fifths spelling with a running tonal centre, seeded by the
KrumhanslSchmuckler key estimate. Pure renderer-side polish — does not
change pitches or durations, just chooses between enharmonic spellings
(e.g., F# vs G♭) to minimise accidental count and stay in-key.

## Procedure

- `humscribe/ensemble/me9_line_of_fifths.py`: line-of-fifths candidate
  generation, key-bonus weighting, leaky-integrator running centre.
- `humscribe/config.py`: `PipelineConfig.enharmonic_spelling = False`
  default, opt-in flag.
- `humscribe/score.py`: `build_stream(enharmonic_spelling=...)` calls
  ME-9 with the inferred KrumhanslSchmuckler key as seed.
- `scripts/eval_me9_spelling.py`: re-build a stream with/without ME-9 from
  the 5 ASAP cached YMT3 predictions, count accidentals on each, verify
  the MIDI pitch sequence is unchanged.

## Results

| piece | acc_baseline | acc_ME9 | drop% | pitches unchanged |
|---|---|---|---|---|
| Bach BWV 854 | 213 | 213 | 0.0 | ✓ |
| Beethoven 21-1 | 54 | 54 | 0.0 | ✓ |
| **Chopin Berceuse** | **96** | **116** | **−20.8** | ✓ |
| Liszt Sonata | 49 | 50 | −2.0 | ✓ |
| Schumann Toccata | 139 | 139 | 0.0 | ✓ |
| **mean** | | | **−4.6%** | ✓ |

Pitches are unchanged on all 5 pieces (correctness verified). But
accidentals **increased** on Chopin Berceuse (96 → 116) and Liszt
Sonata (49 → 50). The mean is a 4.6% regression.

## Interpretation

Two failure modes show up:

1. **KrumhanslSchmuckler picks the wrong key** on the Chopin and Liszt
   cached YMT3 outputs (probably because the pitch distribution within
   the first 30 s of the piece is dominated by non-tonic harmony — common
   in Romantic pieces with delayed tonic resolution). My ME-9 with
   `key_bonus=3.5` then aggressively biases toward "in-key" spellings
   relative to the *wrong* key, producing more accidentals not fewer.

2. **ME-9 has no global accidental budget**. It picks each note's
   spelling locally (running centre + key-bonus), with no rollback when
   the chosen spelling adds an accidental that the alternative wouldn't
   have. A true Temperley implementation has a global-optimum cost
   function across all notes — that's harder and would need a small DP.

## Decision

**Keep ME-9 behind the flag** `PipelineConfig.enharmonic_spelling=False`
(it's not on by default in production). Document this as a negative result
at the current tuning. Two paths forward (Phase F candidates):

- **A**: Use the GT key from the score (where available) instead of
  KrumhanslSchmuckler-from-transcription. On the rendered demos this is
  always known.
- **B**: Implement a true Temperley DP variant that minimises a global
  accidental + voice-leading cost.

Neither blocks the current Phase E loop. The metric pass criterion
(MV2H ≥ +0.01) was never expected to be met by a pitch-spelling member —
ME-9's intended value is reader-visual, which the metric can't see.

## Next

Move on to ME-4 (tonal-meter prior on DP) and ME-7 (anacrusis), which
target the real metric gaps. ME-9 retried as part of Phase F with the
fixes above.
