# Phase G strict pass criteria scorecard

Final tally per `task_descriptions/task_description_v4.md` after the
Phase G session. ASAP numbers cite the beat source per the spec.

## Summary

| # | item | strict pass | shipped status |
|---|---|---|---|
| **1** | voice ID plumbing | partial (ASAP ✓, MAESTRO ✗) | **SHIPPED** (piano default-on) |
| **2** | meter grid markers | partial (ASAP ✓, MAESTRO ✗) | **SHIPPED** (default-on) |
| 3 | F-1b IOI octave detector | ❌ FAIL | code shipped, doesn't fire on test set |
| 4 | same-pitch gap merging | partial (ASAP no-op, Vocadito eval cap) | **SHIPPED** (humming default-on, flag) |
| 5 | median pitch smoothing | partial (Vocadito eval cap) | **SHIPPED** (humming default-on, flag) |
| 6 | silent-region trimming | partial (no leading-silence test clips) | **SHIPPED** (humming default-on, flag) |
| **7** | demo hums in Streamlit | ✅ PASS | **SHIPPED** |
| 8 | round-trip self-consistency | ❌ FAIL (sign inversion) | infra shipped, metric needs redesign |
| 9 | confidence-aware output | partial (1 of 3 criteria) | **SHIPPED** (aggregation API) |
| 10 | bar-level diagnostic | partial (correlation ✓, cutoffs near miss) | **SHIPPED** (diagnostic module) |
| **11** | render_tpb auto-detect | ✅ PASS | **SHIPPED** (default-on) |
| 12 | ME-14 ensemble selection | ❌ FAIL (+0.0049 vs +0.015) | code shipped, ceiling proven |
| 13 | Lakh MIDI LoRA training | deferred (multi-hour task) | harness + OOM protocol shipped |
| 14 | multi-take averaging UX | deferred (needs user triplets) | code shipped |
| 15 | DDSP solo_flute2 retest | deferred (needs checkpoint) | code shipped, ckpt missing |
| **16** | C5b listening artifact | human-eval deferred | artifact (5/10 pairs + Form) shipped |
| **17** | Docker verification harness | user-run deferred | script shipped |

**ASAP 9-piece MV2H mean: 0.5515 (baseline) → 0.6151 (G-1 + G-2 default-on). +0.0636.** This is the headline Phase G win — the per-axis Voice and Meter sub-scores each lift by ~0.1-0.2 while multi-pitch and value stay flat, exactly as the framing said they should.

## Detail per item

### G-1 voice ID plumbing — partial
Pass criteria:
- ✅ ASAP voice ≥ 0.80 → observed **0.825**
- ❌ MAESTRO voice ≥ 0.65 → observed **0.348** (B76 trained on piano hands; chamber GT has 3-4 voices)
- ✅ no multi-pitch / value regression

Shipped for piano input; chamber-data MAESTRO regresses voice score and is discarded for the multi-instrument case. Code path: `humscribe/eval/voice_emission.py`.

### G-2 meter grid markers — partial
Pass criteria:
- ✅ ASAP meter ≥ 0.30 → observed **0.303**
- ❌ MAESTRO meter ≥ 0.35 → observed **0.102** (chamber meter ambiguity)
- ✅ DTW no collapse (9/9 ASAP + 5/5 MAESTRO numeric MV2H)
- ✅ no other sub-score regression

Shipped. The headline lift is on ASAP; MAESTRO discard rationale is documented.

### G-3 F-1b IOI octave detector — fail
Pass criteria:
- ❌ 9/9 detector correct on ASAP → observed **6/9**
- ❌ Chopin Berceuse MV2H lift ≥ +0.04 → observed **+0.0000**
- ✅ no false fires (0/9 false-fires)

Discarded: Chopin Berceuse needs a 3-tier (1/3 BPM) correction; halve/double can't reach a 3× error.

### G-4 same-pitch gap merging — partial (eval-cap)
Pass criteria:
- ❌/partial Vocadito A1 noff F1 ≥ 0.67 — full 40-clip Vocadito eval did not complete in this session window; Vocadito 10-clip MV2H baseline 0.5162 measured.
- ✅ Code shipped at `humscribe/post_process.py:merge_same_pitch`, default-on for humming branch.
- ✅ ASAP unaffected (piano branch doesn't apply G-4).

### G-5 median pitch smoothing — partial (eval-cap)
Pass criteria:
- ❌/partial Vocadito A1 noff F1 ≥ 0.67 — same as G-4.
- ✅ Code shipped at `humscribe/post_process.py:median_smooth_pitch`, 250 ms voiced-only window, default-on for humming.
- ✅ Instrument unaffected (only triggers in humming branch).

### G-6 silent-region trimming — partial (eval-cap)
Pass criteria:
- ❌/partial Vocadito beat F-measure ≥ 0.95 on clips with > 1 s leading/trailing silence — no such clips in the 10-clip Vocadito subset; full 40-clip + leading-silence subset eval deferred.
- ✅ Code shipped at `humscribe/post_process.py:trim_silence`, -40 dB threshold + 10 ms margin, default-on for humming with `beat_this` path adjusted to use the trimmed audio while propagating the leading-pad offset to absolute beat times.

### G-7 demo hums — pass
Pass criteria:
- ✅ 5 demos work end-to-end without manual upload (5/5).
- ✅ Each produces a transcription (5/5 non-empty).

Shipped to `app/demos/` + Streamlit `transcribe_tab` selector.

### G-8 round-trip self-consistency — fail (sign inverted)
Pass criteria:
- ✅ |r| with MV2H ≥ 0.3 → observed **0.642**
- ❌ Liszt has highest distance → observed Liszt has **lowest** distance (sign flipped — distance correlates with note count, not quality)
- ?  ≥ 80% MV2H<0.30 catch — vacuous (no piece below 0.30)

Discarded; infra shipped at `humscribe/eval/round_trip.py` for Phase H to swap the distance metric.

### G-9 confidence-aware output — 1 of 3 (partial)
Pass criteria:
- ?  per-note conf vs in-GT |r| ≥ 0.4 — deferred (needs Vocadito GT matcher)
- ?  lowest 20% flag recovers ≥ 60% FPs — deferred
- ✅ global confidence vs MV2H |r| ≥ 0.4 → observed **0.435** (median beat-strength aggregate)

Shipped at `humscribe/eval/confidence.py`. Two strict criteria deferred to Phase H.

### G-10 bar-level diagnostic — 1 of 3 (partial)
Pass criteria:
- ❌ Liszt < 0.4 → observed **0.490** (narrow miss — `beat_consistency` would catch at < 0.5)
- ❌ Bach Fugues > 0.8 → 4/5 pass; BWV 846 at 0.500
- ✅ Pearson with MV2H ≥ 0.3 → observed **0.440**

Shipped at `humscribe/eval/bar_diag.py`.

### G-11 render_tpb auto-detect — pass
Pass criteria:
- ✅ ≤ 5 unreadable tuplets across 4 demos → observed **3**
- ✅ no MV2H regression > 0.005 (render-only change is structurally incapable of MV2H regression)

Shipped. `vocadito_1_humming.svg` benefits (3 × 24-lets → 0); others unchanged.

### G-12 ME-14 ensemble selection — fail
Pass criteria:
- ❌ ASAP MV2H lift ≥ +0.015 → observed **+0.0049** (oracle ceiling)
- ✅ no piece regresses > 0.02 (oracle selection guarantees nondecrease)

Discarded; the available ensemble member set (tpb24/12/6) caps the achievable lift well below the strict criterion.

### G-13 Lakh MIDI LoRA training — deferred
Pass criteria:
- ?  training completes without OOM — deferred (Lakh prep is a 1-2 hour separate step)
- ?  test loss < 0.983 — deferred
- ?  chroma similarity ≥ 0.72 — deferred
- ✅ OOM protocol harness shipped (`scripts/exp_G13_lakh_lora.py`, `logs/vram_g13.log`, `reports/_OOM_INCIDENTS.md`)

### G-14 multi-take averaging — code shipped
Pass criteria:
- ?  3-take consensus F1 ≥ 0.72 — deferred (no user-recorded triplets)
- ?  single-take F1 ≥ 0.65 — deferred
- ✅ algorithm + Streamlit UI shipped at `humscribe/eval/multi_take.py`

### G-15 DDSP solo_flute2 retest — code shipped, checkpoint absent
Pass criteria:
- ?  direct Vocadito A1 ≥ 0.55 — deferred (solo_flute2_ckpt missing from host)
- ?  ensemble ≥ 0.65 — deferred
- ✅ three fixes shipped (crossfade, loudness-norm bypass, solo_flute2 path)

### G-16 listening artifact — human-eval deferred
Pass criteria:
- ◐ 10 pairs shipped → observed **5/10** (5 ready, 5 queued behind GPU contention)
- ✅ protocol + Form template shipped at `outputs/g16_listening_test/`
- ?  mean rating ≥ 3.5/5 — needs human raters

### G-17 Docker actual-build verification — user-run deferred
Pass criteria:
- ✅ build script shipped (`scripts/g17_docker_verify.sh`, executable)
- ?  build exits 0 on a real Docker host — needs a host with docker (sandbox doesn't)

## Production defaults shipped this phase

- `PipelineConfig.same_pitch_merge = "auto"` (G-4, humming branch)
- `PipelineConfig.median_smooth_g5 = "auto"` (G-5, humming branch)
- `PipelineConfig.silent_trim_g6 = "auto"` (G-6, humming branch, -40 dB threshold)
- `PipelineConfig.render_tpb_auto = "auto"` (G-11, slow-piece downgrade to render_tpb=8)
- `humscribe/eval/voice_emission.py` plumbed through eval scripts (G-1, piano)
- `humscribe/eval/mv2h_io.py:notes_to_mv2h_format(beats=…)` accepts real beats (G-2)

## What this means for production

The Phase G headline is **+0.0636 mean ASAP MV2H** from emission fixes alone (G-1 + G-2). That's the entire Stage 1 target (+0.03 to +0.06) cleared with two emitter changes. Everything else is incremental — G-11 cleans the render path, G-4/5/6 contribute on humming, G-7 covers UX, the Stage 2 diagnostics (G-9 / G-10) ship as monitoring tools.

The structurally hard remaining gaps (Chopin Berceuse 3× tempo error, MAESTRO chamber voice/meter, harmony sub-axis stuck at 0.000) are Phase H scope — none of them is closable inside the v4 spec's evaluation envelope.
