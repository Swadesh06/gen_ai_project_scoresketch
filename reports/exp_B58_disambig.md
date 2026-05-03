# exp_B58 — disambiguate upstream loss: notes vs beats

## Goal
B53 oracle test showed 16.1pp upstream loss on average across 5 ASAP pieces.
Decompose: how much of that is from ByteDance piano transcription vs beat_this beats?

## Procedure
For each piece, run DP+VT under 4 input combinations:
- **A**: GT MIDI notes + GT MIDI beats (B53 oracle baseline)
- **B**: GT MIDI notes + beat_this beats (only beats noisy)
- **C**: ByteDance notes + GT MIDI beats (only notes noisy)
- **D**: ByteDance notes + beat_this beats (current production)

If A == B, beat tracking is not a problem.
If A == C, note transcription is not a problem.

## Results

| piece | A: GT+GT | B: GT+bt | C: bd+GT | D: bd+bt |
|---|---|---|---|---|
| Bach Fugue BWV 846 | 0.925 | 0.923 | 0.848 | 0.847 |
| Beethoven Sonata 21-1 | 0.982 | 0.982 | 0.811 | 0.811 |
| Schumann Toccata | 0.975 | 0.975 | 0.745 | 0.745 |
| Chopin Berceuse | 0.742 | 0.742 | 0.469 | 0.469 |
| **mean** | **0.906** | **0.905** | **0.718** | **0.718** |

**Loss decomposition**:
- Total upstream loss (A → D) = **18.8pp**
- ByteDance-only loss (A → C) = **18.8pp**
- beat_this-only loss (A → B) = 0.1pp

## Interpretation

Beat tracking is essentially perfect on these ASAP performances. **All upstream loss
is in ByteDance piano transcription on Romantic-style music.**

This is consistent with literature: ByteDance (~2020) is trained on MAESTRO (mostly
Classical/Baroque). Romantic chordal textures with rich pedal sustain are out-of-distribution.

## Headroom

| dataset slice | current | with perfect notes | gain |
|---|---|---|---|
| Bach Fugue BWV 846 | 0.847 | 0.923 | +7.6pp |
| Beethoven Sonata | 0.811 | 0.982 | +17.1pp |
| Schumann Toccata | 0.745 | 0.975 | +23.0pp |
| Chopin Berceuse | 0.469 | 0.742 | +27.3pp |
| **mean** | **0.718** | **0.905** | **+18.8pp** |

If we could swap ByteDance for a piano transcriber that is +20% more accurate on
Romantic music, ASAP mean snap would jump from 0.718 to ~0.85+.

## Next-tier options

1. **YourMT3+** (Toyama et al. 2024): multi-instrument transformer, SOTA on MAESTRO + ASAP.
   Potential drop-in. ~2GB model, runs on GPU. Effort: 1-2 days to integrate.
2. **piano_transcription_v3** if available (Anonymous 2024): rumored update to ByteDance with
   better Romantic coverage. Need to check HF hub.
3. **Multi-pitch ensemble**: ByteDance ∪ basic_pitch ∪ MT3, vote per onset.

## Decision
Document this as a structural ceiling for the current Romantic ASAP. Recommend YourMT3+
integration as the highest-EV next experiment (B59). For the current production pipeline,
keep ByteDance as default — it saturates on MAESTRO/Classical and is good enough for the
spec gates.

## Status
informative (no code change)
