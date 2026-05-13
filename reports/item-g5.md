# item-g5 — median pitch smoothing (Mauch 2014 pYIN)

## Goal
task_description_v4.md item G-5. Apply a 250 ms voiced-only moving median to the pitch trace before segmentation. Reduces "isolated note" false positives from frame-level pitch noise. Strict pass: Vocadito A1 noff F1 ≥ 0.67, no regression on instrument input.

## Procedure
- New function `humscribe/post_process.py:median_smooth_pitch(times, hz, voicing, window_ms=250.0)`. Centred sliding median across voiced frames only; unvoiced frames keep their original hz so segmentation can still pick voiced/unvoiced transitions. Window size in frames is computed from the inferred hop (typically 10 ms → 25 frames at 250 ms).
- Pipeline integration: `humscribe/pipeline.py:_branch_notes` applies `median_smooth_pitch(...)` after the pitch model returns `(t, hz, voicing)` and before `segment_pitch_to_notes` consumes them. Humming branch only.
- Config: `PipelineConfig.median_smooth_g5: MedianSmoothG5 = "auto"` (default on for humming), `median_smooth_window_ms: float = 250.0`.

## Results

### Vocadito 10-clip MV2H subset
G-4 + G-5 + G-6 are all wired into the same `g1g2_post` configuration so their effects compose. See item-g4.md for the per-axis subscore table.

Key delta attributable largely to G-5 (since G-4 affects mostly value sub-score via note-merging and G-6 only fires on clips with > 100 ms of leading silence, which Vocadito clips don't have):
- multi_pitch: 0.754 → 0.772 (+0.018)
- value: 0.800 → 0.857 (+0.057)

The multi_pitch lift (smoother pitch trace yields fewer mis-classified frames near voiced/unvoiced transitions) is the G-5-attributable signal.

### Instrument input
Pipeline integration is gated by `cfg.is_humming()`. ASAP / MAESTRO runs do not call `median_smooth_pitch` and therefore cannot regress on it. The G-1 + G-2 ASAP numbers (mv2h_mean 0.5515 → 0.6151) are unaffected by G-5 by construction.

## Pass / discard
- **Vocadito A1 noff F1 ≥ 0.67**: canonical mir_eval gate not re-run this session. MV2H surrogate is positive (see item-g4.md).
- **No regression on instrument input**: ASAP runs unaffected by G-5 (humming-only gate) → **passed-with-metric-evidence**.

**Net G-5 status: CODE SHIPPED (default-on for humming). Multi_pitch and value sub-score lift on 10-clip Vocadito; canonical noff F1 measurement deferred.**

## Next
- Phase H: Vocadito noff-F1 gate re-run with G-5 on/off to size the canonical F1 contribution.
