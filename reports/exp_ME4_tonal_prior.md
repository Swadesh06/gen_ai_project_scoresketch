# ME-4 — tonal-meter prior on the DP — discard at this tuning (ASAP)

## Goal

Phase E item 7 ME-4 per `task_description_v3.md`. Build a P(scale_degree |
beat-position-in-bar) prior from a small MusicXML corpus and use it as a
tie-breaker on the Cemgil-Kappen rhythm DP. Music theory predicts strong
scale degrees (tonic, dominant) on strong beats; the prior captures that.

## Procedure

- `humscribe/ensemble/me4_tonal_meter_prior.py`: walks all Bach four-part
  chorales in music21's corpus iterator (200 pieces, 45,966 notes). For
  each note records (beat-position-in-bar, scale-degree-relative-to-tonic).
  Normalised to a 12×12 conditional probability table, cached to
  `/workspace/.cache/me4_tonal_prior.npz`.
- `scripts/eval_me4_tonal_prior.py`: applies the prior as a post-DP shift —
  for each note's current tatum position, compares log P(degree | pos) with
  ±1 alternative positions and shifts when the gain × λ > DP-cost of the
  shift. Compares MV2H against the baseline (no prior).

## Results (lambda=2.0, tpb=24, 5 ASAP pieces, eval_seconds=30)

| piece | base MV2H | ME-4 MV2H | Δ |
|---|---|---|---|
| Bach BWV 854 | 0.589 | 0.585 | −0.004 |
| Beethoven 21-1 | 0.505 | 0.502 | −0.003 |
| Chopin Berceuse | 0.533 | 0.517 | −0.016 |
| Liszt Sonata | 0.475 | 0.475 | −0.001 |
| Schumann Toccata | 0.497 | 0.493 | −0.005 |
| **mean** | **0.520** | **0.514** | **−0.006** |

All 5 pieces regress (mean −0.006). Decision: **discard at this tuning**.

## Interpretation

The pattern matches ME-9: the cached YMT3 transcriptions on score-rendered
ASAP audio already have very accurate onsets (snap to a few-ms scale). The
tonal prior, applied as a "shift candidate position by ±1 tatum when prior
score is higher" rule, *moves notes away from their accurate timings* to
fit the chorale-derived prior. Net regression.

The use case where ME-4 *should* help is humming (Vocadito) or a noisy
piano transcription where the DP timing is uncertain to begin with. On
clean instrument inputs, the prior fights the (already correct) DP.

## Decision

Discard at this tuning. Phase F retry:

- Evaluate on Vocadito specifically (prior should give more value when
  notes have rubato).
- Train a humming-specific prior (the chorale prior over-fits Baroque
  tonality which may not match Romantic or pop melodic shapes).
- Reduce lambda — the eval used λ=2.0, but on Vocadito a smaller λ
  (e.g., 0.3) may produce gentler shifts that help on uncertain notes
  without breaking certain notes.

## Next

Move on to ME-7 (anacrusis detection) — different failure mode, lighter-
touch correction. ME-11 (formant-band onset detector) targets the
Vocadito offset gap, which is the real headline humming weakness.
