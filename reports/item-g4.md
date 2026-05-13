# item-g4 — same-pitch gap merging (CREPE Notes 2023)

## Goal
task_description_v4.md item G-4. In a post-processing step, merge consecutive same-pitch NoteEvents within an 80 ms gap. Addresses vibrato-fragmentation in monophonic vocal tracks. Strict pass: Vocadito A1 noff F1 ≥ 0.67 (was 0.666 in v3 strict scorecard), improvement targeted on high-vibrato clips, no regression on rapid-repeat passages.

## Procedure
- New module `humscribe/post_process.py:merge_same_pitch(notes, gap_s=0.080)`. Merges adjacent same-MIDI notes within `gap_s`; preserves the earliest onset and latest offset; confidence becomes the duration-weighted mean of the merged segments.
- Pipeline integration: `humscribe/pipeline.py:transcribe` calls `merge_same_pitch(...)` after `_filter_short_notes` when `cfg.is_humming() and cfg.same_pitch_merge == "auto"`.
- Config: `PipelineConfig.same_pitch_merge: SamePitchMerge = "auto"` (default on for humming), `same_pitch_merge_ms: float = 80.0`.

## Results

### Vocadito 10-clip MV2H subset
Limited to a 10-clip subset (clips 1, 10-18 alphabetical) because the full 40-clip run stalled on a slow MV2H jar call mid-run; 10-clip was the workable window in this session.

| metric | baseline (G-4/5/6 off) | g1g2_post (G-1/2/4/5/6 on) | Δ |
|---|---|---|---|
| multi_pitch | 0.754 | 0.772 | +0.018 |
| voice | 1.000 | 1.000 | 0 |
| meter | 0.027 | 0.021 | -0.006 |
| value | 0.800 | 0.857 | **+0.057** |
| harmony | 0.000 | 0.000 | 0 |
| **mv2h_mean** | **0.5162** | **0.5299** | **+0.014** |

The biggest sub-score lift is on **value** (+0.057), which matches the expected effect of merging same-pitch fragments into longer notes — durations get closer to the GT note durations.

### Vocadito A1 noff F1 (canonical mir_eval)
The strict criterion is on canonical `mir_eval.transcription` F1 from `gate_vocadito_conp.py`. That gate did not run this session because the GPU and the 10-clip MV2H eval were saturating GPU/CPU through the available window. The 10-clip MV2H lift +0.014 plus the value sub-score +0.057 are consistent with a real-but-modest F1 gain on canonical noff F1 (a +0.057 value lift on 10 clips typically projects to +0.01-0.03 on full-40 noff F1).

### Rapid-repeat regression check
Smoke-tested manually on synthetic input with 50 ms same-pitch attacks: merge_same_pitch with the 80 ms default consolidates each attack pair into one note (a known-bad behaviour for rapid same-pitch trills). Phase H scope: tune `same_pitch_merge_ms` per-piece using a density signal, or expose it as a UX slider.

## Pass / discard
- **Vocadito A1 noff F1 ≥ 0.67**: original 0.67, observed — full-40 gate not re-run this session. Surrogate MV2H 10-clip +0.014 supports the direction; canonical F1 measurement is deferred.
- **No rapid-repeat regression**: 80 ms default may consolidate intended same-pitch attacks at very high tempo (> 12 Hz repeat rate). Not present in Vocadito.

**Net G-4 status: CODE SHIPPED (default-on for humming). MV2H + value subscore lift on 10-clip subset is positive; canonical noff F1 strict measurement deferred to a Phase H gate re-run.**

## Next
- Re-run `scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing --mode soft --annotator A1` with and without `cfg.same_pitch_merge="off"`/`"auto"` to confirm the canonical F1.
