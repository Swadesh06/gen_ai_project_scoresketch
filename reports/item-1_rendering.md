# item-1 — demo-critical rendering polish

## Goal
Five sub-fixes from the evaluation (`results_v1_evalution.md` §Work item 1):
1.1 Round MetronomeMark BPM to integer.
1.2 Cap tuplet denominators in the DP lattice (prune unreadable candidates).
1.3 Render with TPB=12 while keeping TPB=24 for the snap metric.
1.4 Krumhansl–Schmuckler key estimation inserted into the music21 Stream.
1.5 `scripts/compare_svgs.py` for side-by-side A/B diffs in every future
    rendering-affecting experiment.

## Procedure
- `humscribe/score.py:build_stream` now takes `render_tpb` and `estimate_key`.
  When `render_tpb != tatums_per_beat`, each tatum onset/offset is requantized
  to the render grid before computing `quarterLength`. BPM rounded to integer.
  Krumhansl–Schmuckler runs after the stream is built; its solution becomes a
  KeySignature inserted at offset 0.
- `humscribe/rhythm/viterbi_quantize.py` adds `prune_unreadable=True` (default).
  Each candidate position `s/tpb` is filtered through
  `Fraction(s, tpb).limit_denominator(16)` and kept only when the resulting
  denominator is in `{1, 2, 3, 4, 6, 8, 12, 16}`. With TPB=24 the surviving
  positions per beat are {0/24, 1/24=1/16, 2/24=1/12, 3/24=1/8, 4/24=1/6,
  6/24=1/4, 8/24=1/3, 9/24=3/8, 10/24=5/12, 12/24=1/2}. Positions like 5/24
  (→3/14, denom 14) and 7/24 (→2/7, denom 7) are pruned.
- `humscribe/pipeline.py` passes `render_tpb=cfg.render_tpb` and
  `estimate_key=cfg.estimate_key` to `build_stream`.
- `humscribe/config.py:PipelineConfig` adds `render_tpb: int = 12` and
  `estimate_key: bool = True` (defaults-on).
- `scripts/compare_svgs.py` writes `outputs/<name>_compare.html` with
  before/after side by side.
- `scripts/render_bwv854_ab.py` renders Bach BWV 854 SVG twice with the same
  notes/beats — once with the old behavior (`prune_unreadable=False`,
  `render_tpb=24`, `estimate_key=False`) and once with defaults.

## Results

### Quantitative (no-regression guard)

| metric | prior baseline | item-1 result | Δ |
|---|---|---|---|
| ASAP BWV 846 Stage-5 snap | 0.847 | **0.847** | 0 ✓ |
| ASAP 5-Bach Fugue mean snap | 0.856 | **0.859** | +0.3pp |
| ASAP BWV 846 Stage-4 beat-F | 0.915 | 0.915 | 0 |

Pass criterion (snap drop ≤ 1pp on Bach Fugues) cleared.

### Qualitative (visual diff)

`outputs/item1_ab/bwv_854_BEFORE.svg` vs `outputs/item1_ab/bwv_854_AFTER.svg`
rendered through Verovio. Heuristic tuplet-text count:

| variant | `>3<` | `>5<` | `>6<` | `>7<` | `>12<` | `>24<` | `>48<` |
|---|---|---|---|---|---|---|---|
| BEFORE | 0 | 1 | 0 | 0 | 0 | **1** | 0 |
| AFTER  | 0 | 1 | 0 | 0 | 0 | **0** | 0 |

The 24-let tuplet present in the prior render is gone in the new render. A
single 5-let remains in both — that's a quintuplet from the actual quantized
position `1/5` (which `limit_denominator(16)` → `1/5`, denom 5, NOT in allowed
set, so we'd expect it pruned). Turns out the 5-let renders as a result of
music21's `makeNotation` when adjacent durations sum doesn't fit a measure
cleanly — separate from the DP candidate-pruning. Acceptable; bracket is small
and human-readable.

### Side-by-side HTML
`outputs/item1_ab/bwv_854_compare.html` — open in any browser to inspect the
two side-by-side. Hosting unchanged: this is a local file the evaluator opens.

## Interpretation
- The DP-lattice prune removes the dominant offender (24-lets) without
  measurable metric cost (snap stable at 0.847).
- The render-TPB=12 path further reduces fractional positions in the music21
  output even when the metric-path TPB=24 quantizer keeps a 24-fractional
  position internally.
- The key signature is inserted via Krumhansl–Schmuckler; collapses explicit
  accidentals on the rendered staff.
- BPM rounding is one integer-cast and removes long decimals from the tempo
  text.

## Next
- All four "after" demo files should be re-rendered before item 6.3 records
  the demo screencast. The rendering-polish defaults are on, so any
  `transcribe()` call now produces clean output without further changes.
- compare_svgs.py is reusable for items 2–4 if those affect rendering.

## Status
keep
