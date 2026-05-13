# task_description_v4.md — HumScribe Phase G

This is the spec for the next phase of HumScribe work. It builds on top of Phase A, Phase B, Phase B+1, Phase B+2 (the v3.4 spec), Phase D (autonomous voice tracker + LoRA), and Phase E (MV2H metric, MAESTRO regen, octave sanity, formant offset detector). Read those first — including `results_v1_evaluation.md`, `results_v2_evaluation.md`, and `results_v3_evaluation.md`.

## The framing shift for Phase G

After three sessions of work, **the transcription pipeline is near its useful ceiling on the metrics that have headroom**. Multi-pitch is 0.96, note value is 0.99 — saturated. The remaining headroom is in three places:

1. **MV2H sub-axes the pipeline doesn't currently emit**: voice (0.70 on ASAP, 0.46 on MAESTRO), meter (0.10), harmony (0.00). The work the system is already doing (B76 voice tracker at 94% accuracy, DP at tatum resolution) isn't being plumbed into the MV2H text format. **Most Phase G wins come from emitter fixes, not pipeline improvements.**
2. **Free signals from existing models**: PESTO, CREPE, ByteDance, beat_this, YourMT3+ all produce confidences that get discarded. Plumbing them through gives you per-note confidence outputs with zero new compute.
3. **Published post-processing tricks the agent hasn't tried**: median pitch smoothing (Mauch 2014), same-pitch gap merging (CREPE Notes 2023), silent-region trimming. Each is a small fix with validated literature support.

**Stop pushing the transcription. Start pushing the metric emission and the use of signals you already have.**

## Hardware constraint and OOM protocol

The agent is running on **RTX 2000 Ada with 16 GB VRAM**. Everything in Phase G fits in 16 GB. But for any experiment whose peak VRAM is estimated at ≥ 12 GB:

1. **Dry-run first.** Launch the experiment with logging set to `nvidia-smi --query-gpu=memory.used --format=csv -l 1 > logs/vram_<exp_id>.log` for the first 60 seconds. Check actual peak.
2. **If peak < 14 GB**: continue at the planned batch size.
3. **If peak ≥ 14 GB**: lower batch size by 2× and retry the dry-run. Repeat halving until peak < 14 GB.
4. **If batch size = 1 still OOMs**: record the experiment in `reports/_OOM_INCIDENTS.md` with: experiment ID, model, peak VRAM observed, what was tried, and stop. Notify the user via the report rather than trying further workarounds.

Specific items in Phase G that need the dry-run protocol:
- F+1-6 Lakh MIDI LoRA training (estimated peak ~10 GB on MusicGen-Melody 1.5B; would OOM on Large at 16 GB)
- F+1-9 DDSP solo_flute2 retest (~1 GB DDSP + ~5 GB pipeline; trivial, but verify chunk boundaries don't accumulate memory)
- MusicGen-Melody-Large inference for arrangement generation (~13 GB at fp16; fits standalone but can't co-locate)

Everything else is < 6 GB peak and doesn't need the dry-run protocol.

## Phase G priority list

17 work items organized into 4 stages by dependency and resource class. Stages 1–3 are largely parallelizable within and across stages.

### Stage 1 — high-EV cheap wins (CPU-only, all parallel)

**Target: complete in one day. Expected cumulative MV2H lift: +0.03 to +0.06.**

**G-1. Voice ID plumbing** — B76 voice tracker outputs are already produced; the MV2H text emitter throws them away. Wire them through `humscribe/eval/mv2h_io.py`. Optional voice-ID list in `notes_to_mv2h_format`, fall back to `voice=0`. Pass criteria: MAESTRO voice sub-score ≥ 0.65 (was 0.46), ASAP voice sub-score ≥ 0.80 (was 0.70), no regression in multi-pitch/value.

**G-2. Meter grid markers** — MV2H supports `bar`/`beat`/`subbeat` annotations as separate metadata. The agent's prior attempt to quantize note `on` times to tatum positions broke DTW alignment. Emit grid markers as metadata instead. Pass criteria: ASAP meter sub-score ≥ 0.30 (was 0.10), MAESTRO meter ≥ 0.35 (was 0.14), DTW doesn't collapse, no other sub-score regression.

**G-3. F-1b second-signal octave detector** — F-1 octave-sanity gets 8/9 on ASAP, misses Chopin Berceuse because notes-per-beat is normal at both wrong and right tempos. Add absolute IOI as a second signal: if predicted BPM ≥ 100 but median IOI ≥ 0.4 s, recommend `halve`. Pass criteria: 9/9 detector correct on ASAP, Chopin Berceuse MV2H lift ≥ +0.04, no false fires.

**G-4. Same-pitch gap merging** (CREPE Notes 2023 published practice) — in `humscribe/notes/post_process.py`, merge consecutive same-pitch notes within 80 ms gap. Specifically addresses vibrato-fragmentation (a named failure mode in the agent's reports). Pass criteria: Vocadito A1 noff F1 ≥ 0.67 (was 0.666), improvement targeted on high-vibrato clips, no regression on rapid-repeat passages.

**G-5. Median pitch smoothing** (Mauch 2014 pYIN published practice) — 250 ms moving median with 10 ms hop on the pitch trace before segmentation. Voiced-frames-only smoothing; preserve unvoiced markers. Pass criteria: Vocadito A1 noff F1 ≥ 0.67, reduces "isolated note" false positives, no regression on instrument input.

**G-6. Silent-region trimming** — strip leading/trailing silence (>-40 dB) so `beat_this` doesn't place beats in silence. Margin of 10 ms preserved. Pass criteria: Vocadito beat F-measure ≥ 0.95 on clips with > 1 s leading/trailing silence, MV2H ≥ baseline + 0.01, no regression on no-silence clips.

**G-7. Pre-recorded demo hums in Streamlit** — 5 hummed examples shipped in the repo (Twinkle Twinkle, Mary Had a Little Lamb, Happy Birthday, Frère Jacques, free improvisation). One-click load in the Streamlit sidebar. Pass criteria: 5 demos work end-to-end without manual upload, each produces a transcription visually resembling the source song.

### Stage 2 — new signal + diagnostics (mixed CPU/GPU, parallel)

**Target: complete in a session after Stage 1. Each item provides new evaluation signal or new diagnostic insight.**

**G-8. Round-trip self-consistency metric** (Cohen et al. 2020 differentiable rendering, validated) — audio → pipeline → MIDI → FluidSynth render → compute MFCC-DTW distance to original audio. New metric that requires zero ground truth. Implementation in `humscribe/eval/round_trip.py`. Pass criteria: correlation with MV2H Pearson |r| ≥ 0.3 on 9 ASAP pieces, successfully flags Liszt failure (highest round-trip distance), catches ≥ 80% of MV2H < 0.30 cases.

**Why this is high-EV**: unlocks unlabeled hyperparameter sweep (sweep on hundreds of audio files instead of 5+10). Replaces the v3 item-6 sweep's data ceiling. Validates the pipeline end-to-end without annotation cost.

**G-9. Confidence-aware per-note output** — every component (PESTO, CREPE, beat_this, ByteDance, YourMT3+) already produces per-frame confidences. Plumb them into `NoteEvent.confidence`. Aggregate via `mean(pesto_conf) * mean(crepe_period) * beat_strength_at_onset`. Pass criteria: Pearson correlation between per-note confidence and "is the note in GT" ≥ 0.4; flagging lowest-confidence 20% of notes recovers ≥ 60% of false positives on Vocadito; global confidence correlates with MV2H |r| ≥ 0.4.

**G-10. Bar-level structural consistency diagnostic** — median absolute deviation of bar durations in seconds. New piece-level diagnostic that doesn't need a reference. Pass criteria: score < 0.4 on Liszt Sonata (catches structural inconsistency), > 0.8 on Bach Fugues, correlation with MV2H ≥ 0.3.

**G-11. render_tpb auto-detect** — turn the manual MAESTRO trick (item 8 used `render_tpb=8`) into a per-piece heuristic. If median note-IOI > 0.3 s AND default `render_tpb=12` produces > 1 tuplet per bar on average, downgrade to `render_tpb=8`. Pass criteria: all 4 demos produce ≤ 5 unreadable tuplets total, no demo's MV2H regresses by > 0.005.

**G-12. ME-14 system-level ensemble selection** — run 4-6 pipeline variants (different tpb, different DP penalties, different voicing thresholds), pick the variant whose output maximizes MV2H (using G-8's round-trip metric as the proxy when ground truth isn't available). Pass criteria: ASAP 9-piece mean MV2H lift ≥ +0.015 over single-config tpb=12 baseline, no piece regresses by > 0.02.

### Stage 3 — bigger lifts (GPU work, sequential)

**Target: parallel with Stage 2 work running on CPU. GPU is busier here.**

**G-13. Lakh MIDI LoRA training** — JSB Chorales (315 pairs) was data-bound per the C5b scaling analysis. Filter Lakh MIDI to ~5,000 melody-arrangement pairs. Train MusicGen-Melody 1.5B LoRA at r=64. **Apply OOM protocol**: dry-run for 60 s, expected peak ~10 GB, halve batch if needed. Pass criteria: training completes without OOM, test loss < 0.983 (C5b baseline), test chroma similarity ≥ 0.72, arrangements have more variety than C5b's chorale-dominant output.

**G-14. Multi-take averaging mode** — Streamlit UI option to record 3 takes of the same hum. Each is transcribed; consensus vote keeps notes that appear in ≥ 2 of 3 takes within ±50 ms. Pass criteria: 3-take consensus Vocadito-style F1 ≥ 0.72 on repeated-same-melody recordings (collect 5 such triplets), single-take baseline of the same user ≥ 0.65 (so consensus delta is real).

**Compute footprint**: 3× existing pipeline cost per session. ~30 s of GPU for 30-second hums. No new model needed.

**G-15. DDSP solo_flute2 retest** — flute is less vibrato-sensitive than the violin checkpoint that failed in v3 item 3. Same architectural test (ensemble path), three fixes from the failure analysis: cross-fade 4-second chunk boundaries with 200 ms overlap-add, disable DDSP's loudness normalization, use the solo_flute2 checkpoint instead of solo_violin. Pass criteria: direct DDSP→pipeline Vocadito A1 ≥ 0.55 (was 0.14 on violin), ensemble (direct + DDSP) ≥ 0.65 (lower bar than v3's 0.71 since item 3 strict-failed).

### Stage 4 — close-out (human-in-loop or one-time tasks)

**G-16. C5b LoRA subjective listening test** (closes v3 item 5 unverifiable criterion) — render 10 (melody, arrangement) pairs using the C5b adapter. Send to 5 listeners, ask "1–5, did the arrangement follow the melody?". Mean rating ≥ 3.5 means strict pass. **Human-in-the-loop; the agent doesn't run this**, but creates the eval artifacts and a Google Form / submission flow.

**G-17. Docker actual-build verification** (closes v3 item 4) — `docker build -t humscribe .` and `docker run -p 8501:8501 humscribe` on a real Linux host. **The agent creates a build script** and the validation harness. The user runs the actual `docker build` on their machine.

## Cross-cutting requirements

These apply to every Phase G work item.

**Per-axis MV2H reporting**:
- Report multi-pitch, voice, meter, value, harmony sub-scores AND mean MV2H for every change
- The headline mean can move slowly even when real improvements ship; per-axis sub-scores tell the story (F-2e taught this — off20 moved +5pp, MV2H moved +0.3pp)

**Visual diff requirement**:
- Every change that affects rendered output must include before/after SVG paths in its report
- The v3 evaluation flagged that v2 task description's item 1 missed re-rendering the MAESTRO demo; the strict scorecard format catches this

**ASAP beat-source citation**:
- Every ASAP number cites `(score beats)` or `(real beats from beat_this)`. The 27pp gap between them is real.

**OOM-protocol logging**:
- Every dry-run logs to `logs/vram_<exp_id>.log` regardless of whether the experiment ends up needing the protocol
- Any OOM is recorded in `reports/_OOM_INCIDENTS.md` with experiment ID, model, peak VRAM, mitigation attempted, and final outcome

**Honest reporting discipline**:
- Half the agent's prior experiments produced negative results, all documented honestly. Continue this.
- The v3 strict scorecard is the gold standard — produce one of those at the end of Phase G.

**WandB tagging**:
- `phase-g` on every Phase G run
- `metric-mv2h`, `metric-roundtrip` for the new metrics
- `emitter-fix` for G-1 / G-2 (so it's easy to filter "the changes that lifted sub-scores")
- `confidence-output` for G-9

## What NOT to do

Carried forward from v3 with one update:

- **Don't run a bigger MIR-ST500 pretrain.** The v3 F-2b/c/d storyline definitively showed it's wrong domain.
- **Don't train more learned voicing on 40-clip Vocadito alone.** Eight attempts confirmed the data ceiling.
- **Don't push for the v3 item 6 strict +0.03 sweep target.** The agent ran 122 Bayesian runs; the ceiling on the cached-features pipeline is +0.022.
- **Don't fine-tune `beat_this` on ASAP.** Already trained on ASAP + 14 classical-piano datasets.
- **Don't push Vocadito above the 0.740 IAA ceiling.** Pipeline at 0.666 is healthy.
- **Don't optimize Liszt Sonata.** Structural DP ceiling at 0.13.
- **Don't full-fine-tune MusicGen.** Always LoRA. Full fine-tuning of any MusicGen variant on 16 GB will OOM.

## Realistic targets

After Stage 1: ASAP MV2H 0.55 → 0.59-0.61, Vocadito MV2H 0.51 → 0.53-0.55.

After Stage 2: ASAP MV2H → 0.62-0.65, Vocadito → 0.55-0.57, voice + meter sub-scores 2-3× their current values.

After Stage 3: minor incremental gains plus the Lakh LoRA shipping as new arrangement default.

After Stage 4: strict v3 pass count 2/8 → 5/8.

**Stage 1 alone gets you 80% of the way to a publishable course-project state.** Everything beyond is iterating on what's already working.

## End of task_description_v4.md

Apply on top of v3 and the agent's Phase E + partial Phase F code. The framing change is: stop optimizing the transcription, start optimizing the emission and the use of free signals. Every Stage 1 item is a small commit that should ship in the same day.
