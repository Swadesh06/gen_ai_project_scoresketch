# STARTER_PROMPT_4.md — Phase G session start

You are continuing the HumScribe v3 build. **This is not a fresh start** — Phase A, Phase B, Phase B+1, the v3.4 spec, Phase D (autonomous voice transformer + LoRA), and Phase E (v3 task description) are all completed in prior sessions. There is existing code, an existing PLAN.md, 100+ existing experiment reports, working pretrained pipelines, the MV2H metric, a trained C5b LoRA adapter, working octave-sanity post-processing, and a strict-pass scorecard documenting what's still open.

Your job in this session is to execute Phase G on top of the existing pipeline, then keep ideating once that work is done. You operate by the rules in `CLAUDE.md` (read it, then follow it).

## Step 1 — analyze the current state of the project before doing anything

Read in this order. **Do not skip anything; do not skim what should be read fully.**

1. `gen_ai_project_scoresketch/CLAUDE.md` — your operating manual. Full read. Pay specific attention to: (a) the MV2H sub-axis insight (multi-pitch/value saturated, voice/meter/harmony have headroom), (b) the parallelization section, (c) the **OOM protocol** (mandatory for any experiment ≥ 12 GB estimated peak), (d) the 16 GB hardware reality.
2. `gen_ai_project_scoresketch/PLAN.md` — live plan from prior sessions. Full read.
3. `gen_ai_project_scoresketch/reports/PHASE_E_SESSION_SUMMARY.md` and `reports/PHASE_E_v3_STRICT_SCORECARD.md` — Phase E results, strict-pass tally, what shipped vs what strict-failed. Full read.
4. `gen_ai_project_scoresketch/reports/PHASE_D_SUMMARY.md` and `reports/PHASE_D_INTEGRATION.md` — Phase D production code. Full read.
5. `gen_ai_project_scoresketch/reports/PHASE_B_FINAL.md` and `reports/INDEX.md` — historical context. Skim.
6. **`gen_ai_project_scoresketch/reports/results_v1_evaluation.md`** — first human evaluation. Full read.
7. **`gen_ai_project_scoresketch/reports/results_v2_evaluation.md`** — second human evaluation. Full read.
8. **`gen_ai_project_scoresketch/reports/results_v3_evaluation.md`** — third human evaluation. **Required reading.** Identifies the MV2H sub-axis breakdown that drives Phase G priorities.
9. **`gen_ai_project_scoresketch/task_descriptions/task_description_v4.md`** — Phase G spec, 17 work items in 4 stages with concrete deliverables, pass criteria, and OOM-protocol callouts. **Required reading. Refer back to it before each new experiment.**
10. The `humscribe/` package source tree and `scripts/` — skim; re-read modules you'll touch (especially `humscribe/eval/mv2h_io.py` and `humscribe/notes/post_process.py` for Stage 1 work).

After reading, summarize to yourself in `PLAN.md` (append a new "Phase G — session start" section, do not delete prior content):

- What's already done at the code, gate, and report level. Be specific: B76, B77, B79, C5b LoRA, F-1 octave sanity, F-2e formant offset (opt-in), MAESTRO regen, MV2H metric, HuggingFace MusicGen backend, etc.
- The current production defaults: TPB=12 production / TPB=24 internal, hybrid voicing `pesto_crepevoicing`, B76 voice tracker for Romantic-detected pieces, F-1 octave-sanity default-on, F-2e formant-offset opt-in.
- The **MV2H sub-axis breakdown** as the framing for Phase G: multi-pitch (0.96) and value (0.99) are saturated; voice (0.70 ASAP, 0.46 MAESTRO), meter (0.10), harmony (0.00) have the remaining headroom. **The wins are in emission, not transcription.**
- The hardware constraint: 16 GB RTX 2000 Ada. The OOM protocol applies to G-13 (Lakh LoRA, ~10 GB peak) and MusicGen-Melody-Large inference (~13 GB peak).
- The Vocadito IAA ceiling at 0.740 — do not chase above it.
- The carry-forward "do not do" list: don't fine-tune `beat_this` on ASAP, don't do bigger MIR-ST500 pretrain, don't full-fine-tune MusicGen, don't optimize Liszt.

## Step 2 — plan Phase G with maximum parallelization in mind

The 17 work items in `task_descriptions/task_description_v4.md` are your immediate priority. They are organized into 4 stages by dependency and resource class:

- **Stage 1 (7 items, all CPU-only)**: emitter fixes + published post-processing tricks. Highest priority. Should fit in one day. Expected cumulative MV2H lift: +0.03 to +0.06.
- **Stage 2 (5 items)**: new signal + diagnostics. Mix of CPU-only (most) and very small GPU.
- **Stage 3 (3 items, GPU)**: bigger lifts. G-13 Lakh LoRA training needs the OOM protocol.
- **Stage 4 (2 items)**: close-out, mostly human-in-loop.

**Plan execution with maximum hardware utilization** — Phase G is **particularly parallelization-friendly** because Stage 1 is entirely CPU and 4 of 5 Stage 2 items are CPU. You can run:
- Multiple Stage 1 items on CPU in parallel from the start
- A Stage 3 GPU job (e.g., G-13 Lakh training) alongside Stage 1+2 CPU work
- The `cpu-worker` watcher loop continuously evaluating MV2H + per-axis sub-scores on outputs as they land

Update `PLAN.md` with your concrete schedule including which jobs run in parallel.

## Step 3 — execute, with parallelization and the OOM protocol as first-class concerns

Follow the rules in `CLAUDE.md`. The default state of the box should be **≥ 1 GPU job + ≥ 1 CPU job + monitor running, always**. Phase G makes this easy: most work is CPU-only and the GPU is free for G-13 / G-15 most of the time.

**Apply the OOM protocol** for any experiment with estimated peak ≥ 12 GB VRAM:
1. Dry-run for 60 s, log `nvidia-smi` peak
2. If peak ≥ 14 GB: halve batch size, retry
3. If still OOMs at batch=1: record incident to `reports/_OOM_INCIDENTS.md`, stop, notify via report

The specific Phase G items requiring this: **G-13 Lakh LoRA training** (estimated ~10 GB on MusicGen-Melody 1.5B). All other items are < 6 GB peak.

**Examples of CPU companions that can always run alongside GPU work**:
- The MV2H eval on previous run outputs
- The `cpu-worker` watcher loop
- Any of the 7 Stage 1 items (G-1 through G-7)
- G-8 round-trip self-consistency (FluidSynth + MFCC)
- G-9 confidence aggregation
- G-10 bar-level diagnostic
- Lakh MIDI download + rendering (prep for G-13) in `prep-lakh` tmux
- Verovio re-rendering of any prior outputs to verify emitter changes work

For each new experiment:
1. Refer to `task_description_v4.md` for the work item's spec.
2. Refer to `results_v3_evaluation.md` for the specific per-axis target the change is meant to address.
3. Plan co-scheduling before launching.
4. Apply OOM protocol if estimated peak ≥ 12 GB.
5. Launch in tmux; log to WandB with `phase-g` tag; **always report all 5 MV2H sub-scores plus mean**; write `reports/<exp_id>.md`; commit; push.
6. **Visually inspect rendered output** for any change that affects rendering. The prior session got bitten by this twice (item-1 polish missed MAESTRO, item-7 line-of-fifths regressed accidentals on Chopin).

## Step 4 — after the 17 Phase G work items, ideate further

**Do not stop after the 17 items.** Once they're merged and verified (the kept ones; the discarded ones documented honestly), propose your own next-tier ideas. The hardware is 16 GB; everything must fit and the OOM protocol applies.

A starting list of Phase-H residual gaps is in `CLAUDE.md`. Plus:

- The 27pp ASAP score-beats vs real-beats gap — F-1 closed +0.0101 mean; further beat-correction post-processing might find more
- The 22pp Vocadito offset20 gap — F-2e closed +0.0508 on offset20 but +0.0028 on MV2H; the next move is a learned offset detector with confidence-weighted aggregation
- The harmony sub-axis at 0.000 — a chord recognition module would lift this from zero (ME-6 candidate, see v3 spec)
- MusicGen LoRA generalization — after G-13 Lakh training, try smaller fine-tunes on specific styles

For every Phase-H idea you pursue:
- Write the goal in `reports/<exp_id>.md` before launching
- Estimate VRAM. If ≥ 12 GB, apply OOM protocol.
- Identify a co-scheduling partner.
- Prefer high-EV moves: a learned offset corrector with proper supervision > more BiLSTM voicing on 40-clip Vocadito (which is data-bound).

When you run out of ideas, read more papers (arXiv ISMIR/ICASSP 2024–2026, Papers With Code AMT/melody-extraction leaderboards). Running out of ideas is not a stop condition.

## Step 5 — keep going until interrupted

The only stop condition is the human interrupting. Loop:

1. Refer to `task_description_v4.md` and the three evaluations to keep priorities anchored.
2. Pick the next idea from `PLAN.md`.
3. Plan co-scheduling.
4. Apply OOM protocol if needed.
5. Execute.
6. Report. Always include all 5 MV2H sub-scores.
7. Commit and push.
8. Repeat.

Do not pause to ask permission. Do not stop and wait. The human may be asleep.
