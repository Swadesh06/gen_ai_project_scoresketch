# STARTER_PROMPT_3.md — Phase E session start

You are continuing the HumScribe v3 build. **This is not a fresh start** — Phase A, Phase B, Phase B+1, the v3.4 spec (`task_description_v2.md`), and the autonomous Phase D work were all completed in prior sessions. There is existing code, an existing PLAN.md, 80+ existing experiment reports, working pretrained pipelines, three trained-from-scratch components (B76 voice transformer, B77 MusicGen LoRA infrastructure, B79 per-voice DP), and passing gates. Do not redo any of that.

Your job in this session is to execute Phase E on top of the existing pipeline, then keep ideating once that work is done. You operate by the rules in `CLAUDE.md` (read it, then follow it).

## Step 1 — analyze the current state of the project before doing anything

Read in this order. **Do not skip anything; do not skim what should be read fully.**

1. `gen_ai_project_scoresketch/CLAUDE.md` — your operating manual. Full read, including the parallelization section, which is critical for hardware utilization in Phase E (most Phase E work items are CPU-only, ideal for parallelization).
2. `gen_ai_project_scoresketch/PLAN.md` — the live plan from prior sessions. Full read.
3. `gen_ai_project_scoresketch/reports/PHASE_D_SUMMARY.md` and `reports/PHASE_D_INTEGRATION.md` — what Phase D shipped (B76 voice transformer, B77 MusicGen LoRA pipeline, B79 per-voice DP, soft-IAA scoring). Full read.
4. `gen_ai_project_scoresketch/reports/PHASE_B_FINAL.md` and `reports/INDEX.md` — historical context. Skim for the chronological view; deep read the Phase B+1 and Phase B+2 sections.
5. **`gen_ai_project_scoresketch/reports/results_v1_evaluation.md`** — first human evaluation of the Phase B+1 results. Full read.
6. **`gen_ai_project_scoresketch/reports/results_v2_evaluation.md`** — second human evaluation, after Phase D. Identifies the 27pp ASAP gap (score-beats vs real-beats), the unfinished MAESTRO file regeneration, the offset20 gap, and the rendering-polish wins. **Required reading.**
7. **`gen_ai_project_scoresketch/task_descriptions/task_description_v3.md`** — Phase E spec, 8 work items with concrete deliverables, pass criteria, dependency graph, and 14 ranked ensemble member candidates. **Required reading. Refer back to it before each new experiment.**
8. `gen_ai_project_scoresketch/task_descriptions/task_description_v2.md` — v3.4 spec, mostly shipped; item 8 (MAESTRO regen) is still open and is Phase E item 8.
9. The `humscribe/` package source tree and `scripts/` — skim; re-read modules you'll touch.

After reading, summarize to yourself in `PLAN.md` (append a new "Phase E — session start" section, do not delete prior content):
- What's already done at the code, gate, and report level (be specific: B76, B77, B79, YourMT3+ default, MusicGen Stage 7, rendering polish on 3 of 4 demos, etc.).
- The current production defaults (TPB=24 internal, TPB=12 render, hybrid voicing pesto_crepevoicing, voice tracking with adaptive_pj on B76 transformer for Chopin-style, per-voice DP for Chopin-style, target_bpm=110 correction).
- The two biggest unfixed weaknesses: **27pp ASAP score-beats vs real-beats gap** (the actual production number is 0.506, not the 0.774 headline) and **22pp Vocadito offset20 gap** (0.439 vs IAA 0.642).
- The **important correction from v2 evaluation**: do NOT attempt to fine-tune `beat_this` on ASAP. It was already trained on ASAP plus 14 classical-piano datasets. The 27pp gap needs algorithmic post-processing or accepting both numbers, not more data.
- The Vocadito IAA ceiling at 0.740 — do not chase above it.
- The **MAESTRO chamber demo file still showing pre-polish output** — picked up as Phase E item 8, one-line fix.

## Step 2 — plan Phase E with maximum parallelization in mind

The eight work items in `task_descriptions/task_description_v3.md` are your immediate priority. They are largely parallelizable; the dependency graph is in the spec. **Plan execution with maximum hardware utilization** — Phase E is particularly parallelization-friendly because most items are CPU-only:

- **Work item 1 (MV2H metric)** is CPU-only and unblocks items 6 and parts of 7. Build it first; it lets every subsequent experiment be evaluated against the new end-to-end objective.
- **Work item 2 (MIR-ST500 stack)** is GPU training (~3 GB VRAM during training). Co-schedule the data download + audio decode on CPU in parallel.
- **Work item 3 (DDSP timbre-transfer)** is small GPU (~1 GB) + CPU eval. Co-schedule with literally anything else.
- **Work item 4 (Docker build)** is CPU-only and long-running. Background job in a dedicated tmux session.
- **Work item 5 (JSB Chorales LoRA)** is GPU-bound (~10 GB during training). Cannot co-locate with another large GPU model but happily co-runs with all CPU work.
- **Work item 6 (MV2H-driven sweep)** is CPU-only after a one-time feature-caching step. Once features are cached, run ~6 parallel sweep agents.
- **Work item 7 (ensemble members)** — 12 of 14 are CPU-only. Ideal filler work. Slot in as the next CPU job whenever the GPU is running solo.
- **Work item 8 (MAESTRO regen)** — one CLI call, drop in anywhere.

Update `PLAN.md` with your concrete schedule including which jobs run in parallel.

## Step 3 — execute, with parallelization as a first-class concern

Follow the rules in `CLAUDE.md`. The default state of the box should be **≥ 1 GPU job + ≥ 1 CPU job + monitor running, always**. If you launch a GPU job and don't have a CPU companion running, that's a bug — find work to fill the CPU. Examples of CPU companions that can always run:

- The MV2H eval (item 1)
- The Docker build (item 4)
- The MV2H-driven sweep agents (item 6, after one-time feature cache)
- Any of the 12 CPU-only ensemble members from item 7
- Re-rendering existing outputs after a rendering change to verify visual cleanup
- Computing additional metrics on cached predictions
- Generating side-by-side SVG diffs for report figures
- Preprocessing the next dataset
- Drafting the next experiment's `reports/<exp_id>.md` skeleton

Co-schedule small GPU jobs together when their combined peak VRAM is under 85% of 32 GB. The MIR-ST500 training (~3 GB) + DDSP timbre-transfer (~1 GB) + a Verovio rendering job + an MV2H sweep agent is a realistic full-saturation configuration.

For each new experiment:
1. Refer to `task_description_v3.md` for the relevant work item's spec.
2. Refer to `results_v2_evaluation.md` for visual/qualitative considerations the prior pass missed.
3. Plan co-scheduling before launching.
4. Launch in tmux; log to WandB with `phase-e` tag; write `reports/<exp_id>.md`; commit; push.
5. **Visually inspect rendered output** for any change that affects rendering — the prior session had cases where the metric improved and the SVG got worse, or where the report claimed a re-render that didn't actually happen.

## Step 4 — after the eight Phase E work items are stable, ideate further

**Do not stop after work items 1–8.** Once they're merged and verified (the kept ones; the discarded ones documented honestly), propose your own next-tier ideas. The hardware is a 32 GB Blackwell with no session caps — be aggressive and ambitious.

A starting list of Phase-F ideas is in `task_description_v3.md` "Future-ideation items" section. Plus the residual gaps you'll have identified by then. Concretely:

- You can fit any single open-weights music model up to MusicGen-Melody-Large (3.3B) at fp16.
- You have headroom to fine-tune small/medium models with LoRA (10–16 GB VRAM, see B77 and Phase E item 5 for the patterns).
- You can run multi-agent WandB sweeps with several parallel workers (Phase E item 6 is the template).
- You can co-locate two medium models on the same card for ablations.
- Always-on means you can leave overnight runs going.

A starting list of Phase-F ideas in `CLAUDE.md` ("Phase F — your own ideas"). Pick from that list, propose your own, or both. For every Phase-F idea you pursue:

- Write the goal in `reports/<exp_id>.md` before launching, so it's not a fishing expedition.
- Estimate VRAM and CPU footprint; identify a co-scheduling partner.
- If the idea is high-risk-high-reward (e.g., a learned beat-correction post-processor for the 27pp gap, a formant-band learned offset detector for the 22pp humming offset gap, LoRA fine-tuning MusicGen on Lakh MIDI), prefer it over yet another hyperparameter sweep on already-tuned components.
- Combine ideas when sensible. A formant-band onset detector trained on MIR-ST500 is a more interesting experiment than either alone.

When you run out of ideas, read more papers (arXiv ISMIR/ICASSP 2024–2025, Papers With Code AMT/melody-extraction leaderboards). Running out of ideas is not a stop condition.

## Step 5 — keep going until interrupted

The only stop condition is the human interrupting. Loop:

1. Refer to `task_description_v3.md` and both evaluations to keep priorities anchored.
2. Pick the next idea from `PLAN.md`.
3. Plan co-scheduling.
4. Execute.
5. Report.
6. Commit and push.
7. Repeat.

Do not pause to ask permission. Do not stop and wait. The human may be asleep.
