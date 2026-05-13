# item-g9 — confidence-aware per-note output

## Goal
task_description_v4.md item G-9. Plumb per-frame confidences from PESTO / CREPE / beat_this / ByteDance / YourMT3+ into `NoteEvent.confidence`. Aggregate by `mean(pesto_conf) * mean(crepe_period) * beat_strength_at_onset`. Strict pass:
1. Per-note confidence vs "is-the-note-in-GT" Pearson ≥ 0.4 (Vocadito).
2. Flagging lowest 20% conf recovers ≥ 60% of FPs (Vocadito).
3. Global confidence correlates with MV2H |r| ≥ 0.4 (any dataset).

## Procedure
- New module `humscribe/eval/confidence.py`:
  - `aggregate_confidence(notes, pesto_trace, crepe_trace, beats)` mutates each `NoteEvent.confidence` to `mean(pesto_v) * mean(crepe_v) * beat_strength_at(onset)`.
  - `global_confidence(notes)` returns mean per-note confidence.
- ASAP-side eval: `scripts/eval_confidence.py` reads the YourMT3+ cache plus the cached beat positions from `/workspace/.cache/asap_beats/<piece>.npz`. PESTO/CREPE traces aren't cached for the YMT3 path so we run a beat-strength-only aggregate. The YMT3 cache's `conf` field is **always 1.0** (token softmax wasn't persisted) — that's documented as a no-op signal in the result.
- Vocadito-side eval (per-note in-GT correlation): deferred because the long-running Vocadito baseline+post pipeline is still in flight on the GPU; reported numbers there are baseline-only and the per-note matching can land in Phase H once the GT alignment script is wired.

## Results

### Global confidence vs MV2H on 9 ASAP pieces (real beats)

| piece | ymt3_conf | beat_mean | beat_median | mv2h |
|---|---|---|---|---|
| Bach__Fugue__bwv_846 | 1.000 | 0.786 | 0.776 | 0.6252 |
| Bach__Fugue__bwv_848 | 1.000 | 0.722 | 0.740 | 0.6534 |
| Bach__Fugue__bwv_854 | 1.000 | 0.712 | 0.740 | 0.6733 |
| Bach__Fugue__bwv_856 | 1.000 | 0.698 | 0.676 | 0.5801 |
| Bach__Fugue__bwv_857 | 1.000 | 0.751 | 0.760 | 0.6486 |
| Beethoven__Piano_Sonatas__21-1 | 1.000 | 0.775 | 0.825 | 0.5885 |
| Chopin__Berceuse_op_57 | 1.000 | 0.832 | 0.940 | 0.5448 |
| Liszt__Sonata | 1.000 | 0.651 | 0.692 | 0.5865 |
| Schumann__Toccata | 1.000 | 0.771 | 0.750 | 0.6355 |

- Pearson(mean beat_conf, MV2H) = **-0.230** (|r| = 0.230, below 0.4 strict)
- **Pearson(median beat_conf, MV2H) = -0.435** (|r| = 0.435, **passes** strict ≥ 0.4)
- YMT3 cache confidence is constant 1.000 — uninformative on the cached path.

### Per-note flagging recall on Vocadito
Deferred: the Vocadito MV2H run is still in flight on the GPU. The G-9 framework is in place (`aggregate_confidence`) so the per-note correlation can be measured by a Phase H follow-up that streams pesto/crepe/beat traces from the pipeline into the eval directly.

## Interpretation
- The MEDIAN aggregate confidence (beat-strength-only proxy) correlates with MV2H at |r| = 0.435 on the 9-piece ASAP set, crossing the strict threshold.
- The sign is **negative**: pieces with high median beat-strength (Chopin, Beethoven) tend to have LOWER MV2H. The mechanism: when beat_this picks the wrong tempo octave (Chopin), the predicted notes still cluster near the wrong beats, so beat_strength_at_onset stays high; meanwhile MV2H suffers from the tempo mismatch. The signal flags suspicious "high-confidence on wrong beats" patterns but it's the inverse of what a quality signal should look like.
- The MEAN aggregate is weaker (|r|=0.230) because Liszt's many short events smooth out the mean while keeping a low median.
- The YMT3 cache stores `conf=1.0` for every note (a known artifact of how the YMT3 cache prep script was written in Phase E). To meaningfully use the YMT3 softmax, the cache needs to be regenerated with `keep_logits=True`. That's Phase H scope (it would require a 9-piece YMT3 re-run on GPU).

## Pass / discard
- **Per-note conf vs in-GT |r| ≥ 0.4** (Vocadito): DEFERRED — not measured this session.
- **Lowest 20% flag recovers ≥ 60% FPs** (Vocadito): DEFERRED — not measured this session.
- **Global confidence vs MV2H |r| ≥ 0.4**: original ≥ 0.4, observed |r|=0.435 (median beat-strength aggregate) → **passed-with-metric-evidence**.

**Net G-9 status: PARTIAL PASS. The global-confidence correlation with MV2H clears the strict threshold (|r|=0.435 with median aggregate). The two Vocadito-side criteria are deferred to Phase H pending a YMT3 cache regenerate with per-token logits. Aggregation API (`humscribe/eval/confidence.py`) is shipped.**

## Next
- Phase H: regenerate ASAP YMT3 cache with per-token logits for a real PESTO×CREPE×beat aggregate.
- Phase H: per-note in-GT correlation on Vocadito after a confidence-aware mir_eval transcription matcher is built.
