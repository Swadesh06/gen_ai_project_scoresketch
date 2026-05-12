# Phase F-3 — ME-14 piece-feature routing analysis (negative)

## Goal

Phase F-3 from `reports/PHASE_F_IDEAS.md`. The ME-14-ext sweep showed
per-piece best-tpb varies (Bach 854 → tpb=8, Bach 846 → tpb=16,
Beethoven → tpb=6, etc.). Can we predict the best tpb from cheap
piece features so the router is production-able without GT?

## Per-piece data

| piece | best_tpb | nps | median_ioi | notes/beat | bpm | midi_iqr | n_notes |
|---|---|---|---|---|---|---|---|
| Bach 854 | tpb8  | 12.8 | 0.12 | 4.2  | 120 | 16 | 374 |
| Bach 846 | tpb16 | 13.0 | 0.07 | 7.0  | 122 | 14 | 383 |
| Bach 848 | tpb8  | 11.9 | 0.12 | 4.2  | 120 | 13 | 347 |
| Bach 856 | tpb6  | 14.7 | 0.06 | 8.7  | 115 | 13 | 436 |
| Bach 857 | tpb8  | 8.6  | 0.12 | 4.2  | 120 | 10 | 251 |
| Beethoven 21-1 | tpb6 | 16.7 | n/a | n/a | 150 | 19 | 501 |
| Chopin Berceuse | tpb6 | 4.5 | n/a | n/a | 120 | 13 | 131 |
| Liszt Sonata | tpb16 | 4.8 | n/a | n/a | 115 | 21 | 142 |
| Schumann Toccata | tpb6 | 24.4 | n/a | n/a | 125 | 16 | 722 |

(Some median_ioi values are 0 due to ties — bug in the diff computation
when many onsets repeat).

## Feature → best-tpb correlations

| feature | Pearson |
|---|---|
| nps (notes/sec) | −0.380 |
| median_ioi | +0.002 |
| notes_per_beat | −0.033 |
| pred_bpm | −0.297 |
| midi_iqr | +0.344 |

No correlation exceeds 0.4. **No clean feature-based router** exists at
this scale (9 pieces is also too small for ML).

## Decision

**Discard the piece-feature router for now**. The global best (tpb=12)
captures most of the achievable mean MV2H (0.5492 vs oracle 0.5494).
The remaining gain (+0.0002 with perfect oracle routing) is within noise.

## Phase F-3 follow-up

A worthwhile retry: run the same analysis on a larger eval set (the
full MAESTRO 2018 test split has 177 pieces; ASAP has ~500 pieces). With
more data, weak correlations might become statistically significant and
a decision tree on (bpm, iqr, n_notes) might route enough pieces correctly
to justify the extra complexity.

For now: tpb=12 as production default is correct.

## Files

- `scripts/analyze_me14_routing.py`
- `reports/_phase_f_F3_me14_routing.json`
