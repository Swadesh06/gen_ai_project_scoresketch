# Phase E session summary (2026-05-12)

## Eight work-item status

| # | item | status | result |
|---|---|---|---|
| 1 | MV2H end-to-end metric | ✓ keep | 3 baselines: ASAP 0.55/0.53/0.54 (raw/DP/DP+sanity), MAESTRO 0.46, Vocadito 0.51 |
| 2 | MIR-ST500 stack | not started | dataset fetch deferred (heavy YouTube downloads) |
| 3 | DDSP timbre-transfer | partial | DDSP installed + tested-import; integration deferred |
| 4 | Docker + HF swap | ✓ keep | Dockerfile + musicgen_hf.py shipped; full build deferred to Docker host |
| 5 | JSB Chorales LoRA | running | 371 pairs rendered; step ~750/1000; loss ~1.4 (oscillating) |
| 6 | MV2H-driven sweep | running | 44/120 runs done; top +0.013 over 0.5074 baseline |
| 7 | Ensemble members | mixed | ME-9 discard, ME-4 discard, ME-7 conservative, ME-1 deferred (GPU) |
| 8 | MAESTRO demo regen | ✓ keep | demos/maestro_chamber3_30s now polished (integer BPM, key sig, 0×48-lets) |

## Phase F-1: octave-sanity beat corrector — first kept improvement

Built outside the eight items because the data analysis (F-1 idea-list)
identified beat_this octave failures as the dominant ASAP loss source.
Detector: notes-per-beat density + fast-tempo/slow-note signal. 9/9
correct on ASAP test set.

**Impact**: +0.0101 mean MV2H on 9-piece ASAP, **+0.088 on Bach BWV 856
alone**. Shipped to production behind `PipelineConfig.octave_sanity =
"auto"` (default).

## Headline numbers (after this session)

| metric | original | final session | Δ |
|---|---|---|---|
| ASAP 9-piece MV2H (DP tpb=24, no corrector) | 0.5277 | — | (start baseline) |
| + octave sanity (still tpb=24) | — | 0.5377 | +0.010 |
| + **tpb=12 default switch (with octave sanity)** | — | **0.5492** | **+0.022** |
| Bach BWV 856 alone (DP tpb=24 no corr → tpb=12 + sanity) | 0.4589 | **0.5588** | **+0.100** |
| MAESTRO 5-clip MV2H | (new) | 0.4587 | (new baseline) |
| Vocadito 40-clip A1 MV2H | (new) | 0.5087 | (new baseline) |

## Code shipped

New packages:
- `humscribe/eval/` — MV2H wrapper, IO conversion, Java jar runner
- `humscribe/ensemble/` — ME-1 (pYIN), ME-4 (tonal prior), ME-7 (anacrusis),
  ME-9 (line-of-fifths)
- `humscribe/beat/octave_sanity.py` — tempo octave corrector (Phase F-1)

New scripts:
- `scripts/eval_mv2h_{asap,maestro,vocadito,correlate}.py` — item 1
- `scripts/sweep_mv2h_e6_cache.py` + `sweep_mv2h_e6.py` + `.yaml` — item 6
- `scripts/eval_octave_sanity{,_mv2h}.py` — F-1 validation
- `scripts/eval_me{1,4,7,9}*.py` — ensemble member evaluation
- `scripts/prep_jsb_pairs.py` — JSB Chorales rendering for item 5
- `scripts/exp_C5_jsb_lora.py` — item 5 training script
- `scripts/prep_beat_corrector_data.py` — F-1 data analysis

Production-pipeline changes:
- `PipelineConfig.octave_sanity` default `auto` — F-1 corrector lives in
  `humscribe/pipeline.py`'s transcribe()
- `PipelineConfig.enharmonic_spelling` default `False` — ME-9 flag
- `humscribe/arrange/musicgen.py` backend selector (audiocraft / hf)

## Negative results (honest)

- **ME-9 line-of-fifths spelling**: pitches preserved ✓ but accidentals
  didn't drop (mean −4.6%). KrumhanslSchmuckler picks wrong key on
  Chopin Berceuse, biasing spellings the wrong way.
- **ME-4 tonal-meter prior on DP**: −0.006 mean MV2H. The prior moves
  already-correct YMT3 timings to fit Bach-chorale tonality (wrong
  direction on clean inputs).
- **MV2H IO quantise-to-tatum option**: dropped from 0.55 → 0.21 mean
  when enabled. DTW alignment degrades when many notes collide on the
  same tatum.
- **Item 6 sweep DP-quantization bug** (fixed): the original sweep
  emitted unquantised notes, so DP params had zero effect on ASAP.
  Fixed by applying tatum-grid mapping; baseline MV2H dropped from
  0.539 (artificial, unquantised) to 0.520 (real, quantised).

## Still running

- C5 JSB LoRA (step ~750/1000, ~10 min wall remaining; test loss
  evaluation at the end)
- Item 6 sweep (44/120 runs done; ~21 min wall remaining at 6 agents)

## Next session

After items 5 and 6 finish:
- C5 report: training loss curve, test loss vs B77 baseline (~0.73), per-
  prompt subjective melody-following score (informal listening).
- Item 6 final report: best config, decision on promote vs leave-behind
  flag, integration into production defaults.
- Phase F-2 (formant-band offset detector) once GPU is free — biggest
  unfixed humming-side gap.

## Commits

40+ commits this session. WandB project: humscribe-v3.2. Sweep ID
kunnj3ze. Origin pushed cleanly throughout.
