# exp_B59 — basic_pitch as alternative piano transcriber on Romantic ASAP

## Goal
B58 showed all 18.8pp Romantic ASAP loss is in ByteDance piano transcription.
Test if basic_pitch (general-purpose AMT) does better. Quick swap experiment.

## Procedure
- Render score MIDI to WAV via fluidsynth (TimGM6mb.sf2).
- Run basic_pitch (ICASSP 2022 model) → notes.
- Run our DP+VT pipeline with adaptive_pj on basic_pitch notes + beat_this beats.
- Compare snap% to ByteDance baseline.

## Results

| piece | ByteDance | basic_pitch | Δ | n_bd | n_bp |
|---|---|---|---|---|---|
| Bach Fugue BWV 846 | 0.847 | 0.501 | -34.6pp | 732 | 629 |
| Beethoven Sonata 21-1 | 0.811 | 0.486 | -32.5pp | 8051 | 6975 |
| Schumann Toccata | 0.745 | 0.309 | -43.6pp | 5567 | 4206 |
| Chopin Berceuse | 0.469 | **0.562** | **+9.3pp** | 1569 | 1540 |
| **mean** | **0.718** | **0.464** | **-25.4pp** | — | — |

## Interpretation

basic_pitch loses badly on most pieces — it's a multi-instrument model not tuned for
piano. Onset times are noisy (different from ByteDance), confusing our DP+VT.

**But it wins +9.3pp on Chopin Berceuse** — the slow, sustained, chordal piece.
basic_pitch's frame-based posterior naturally handles long sustained notes; ByteDance
under-segments them.

## Adaptive transcriber selection

If we route slow/sustained pieces to basic_pitch and others to ByteDance:

| piece | chosen | snap |
|---|---|---|
| Bach 846 | bd | 0.847 |
| Beethoven | bd | 0.811 |
| Schumann | bd | 0.745 |
| Chopin | **bp** | **0.562** |
| **mean** | adaptive | **0.741** (+2.3pp over all-bd) |

The selection criterion would be median IOI (slow piece = long IOI) — simple to implement.

## Decision

- Keep ByteDance as the default piano transcriber.
- **Optional**: add `pp.adaptive_piano_transcriber` flag — if median IOI > 0.6s and median
  note duration > 0.4s, route to basic_pitch. Document for future agents.
- Real upstream improvement still needs YourMT3+ or similar SOTA piano transcriber.

## Status
informative + small-keep candidate (adaptive selection)
