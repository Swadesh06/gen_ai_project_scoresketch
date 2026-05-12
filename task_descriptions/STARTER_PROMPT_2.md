# STARTER_PROMPT_2.md — Phase B+2 session start

You are continuing the HumScribe v3 build. **This is not a fresh start** — Phase A and Phase B+1 were completed in a previous session. There is existing code, an existing PLAN.md, 30+ existing experiment reports, working pretrained pipelines, and passing gates. Do not redo any of that.

Your job in this session is to execute the next phase of work on top of the existing pipeline, then keep ideating once that work is done. You operate by the rules in `CLAUDE.md` (read it, then follow it).

## Step 1 — analyze the current state of the project before doing anything

Read in this order. **Do not skip anything; do not skim what should be read fully.**

1. `gen_ai_project_scoresketch/CLAUDE.md` — your operating manual. Full read, including the parallelization section, which is critical for hardware utilization.
2. `gen_ai_project_scoresketch/PLAN.md` — the live plan from the previous session. Full read.
3. `gen_ai_project_scoresketch/reports/PHASE_B_FINAL.md` and `reports/PHASE_B_SUMMARY.md` — what was tried, what won, what was discarded with rationale. Full read.
4. `gen_ai_project_scoresketch/reports/INDEX.md` — chronological index of every prior experiment.
5. **`gen_ai_project_scoresketch/reports/results_v1_evaluation.md`** — the human's evaluation of the Phase B+1 results, including visual analysis of rendered SVGs. **This is required reading. It identifies issues the metrics did not catch.**
6. **`gen_ai_project_scoresketch/task_descriptions/task_description_v2.md`** — the spec for what to do next, with six concrete work items, success criteria, and a dependency graph. **This is required reading. Refer back to it before each new experiment.**
7. The `humscribe/` package source tree and `scripts/` — skim, then re-read modules you're about to touch.

After reading, summarise to yourself in `PLAN.md` (append a new "Phase B+2 — session start" section, do not delete prior content):
- What's already done at the code, gate, and report level.
- The current production defaults (TPB=24, hybrid voicing pesto_crepevoicing, voice tracking with adaptive_pj, etc.).
- The two biggest unfixed weaknesses identified by the evaluation: rendered-SVG over-complexity (24-lets, 48-lets) and the offset-F1 gap on humming (0.439 vs human IAA 0.642).
- The B58 finding: 100% of remaining ASAP loss is in ByteDance, beat tracking and DP are essentially perfect.
- The Vocadito IAA ceiling at 0.740 — do not chase above it.

## Step 2 — plan the work

The six work items in `task_descriptions/task_description_v2.md` are your immediate priority. They are largely parallelizable; the dependency graph is in the spec. **Plan their execution with maximum hardware utilization in mind**:

- Work item 1 (rendering polish) is mostly CPU work. Run it alongside any GPU work.
- Work item 2 (YourMT3+ integration) is GPU inference on a new model. Co-schedule the model download + smoke test with CPU-bound work.
- Work item 3 (MusicGen-Melody arrangement) is GPU inference on a different new model. The model load is heavy (~13 GB VRAM), so do not co-schedule another large GPU model during arrangement runs — but absolutely do run CPU work in parallel.
- Work item 4 (voicing exit hysteresis) is a CPU-bound sweep on cached pitch traces. Perfect filler while GPU jobs run.
- Work item 5 (MedleyDB pseudo-labeling) is speculative; skip if the others are running long.
- Work item 6 (final polish) consolidates everything; dependencies on 1–4 to land first.

Update `PLAN.md` with your concrete schedule including which jobs run in parallel.

## Step 3 — execute, with parallelization as a first-class concern

Follow the rules in `CLAUDE.md`. The default state of the box should be **≥ 1 GPU job + ≥ 1 CPU job + monitor running, always**. If you launch a GPU job and don't have a CPU companion running, that's a bug — find work to fill the CPU. Examples of CPU companions that can always run:

- Re-rendering existing outputs after a rendering-polish change to verify visual cleanup
- Computing additional metrics (offset-F1, beat-F at multiple tolerances) on cached predictions
- Generating side-by-side SVG diffs for the report figures
- Preprocessing the next dataset
- Drafting the next experiment's `reports/<exp_id>.md` skeleton

Co-schedule small GPU jobs together when their combined peak VRAM is under 85% of 32 GB. PESTO + CREPE + ByteDance combined is well under 10 GB; you can run several independent humming-pipeline experiments on the same card.

For each new experiment:
1. Refer to `task_description_v2.md` for the relevant work item's spec.
2. Refer to `results_v1_evaluation.md` for any visual or qualitative considerations the agent's prior pass missed.
3. Plan co-scheduling before launching.
4. Launch in tmux; log to WandB; write `reports/<exp_id>.md`; commit; push.
5. **Visually inspect rendered output** for any change that affects rendering — the prior session had cases where the metric improved and the SVG got worse. Don't rely on metrics alone for rendering changes.

## Step 4 — after the six work items are stable, ideate further

**Do not stop after work items 1–6.** Once they're merged and verified, propose your own next-tier ideas. The hardware is a 32 GB Blackwell with no session caps — be aggressive and ambitious. Concretely:

- You can fit any single open-weights music model up to MusicGen-Melody-Large (3.3B) at fp16.
- You have headroom to fine-tune small/medium models with LoRA (10–16 GB VRAM, see `audiocraft/musicgen-dreamboothing` for the pattern).
- You can run multi-agent WandB sweeps with several parallel workers.
- You can co-locate two medium models on the same card for ablations.
- Always-on means you can leave overnight runs going without session-cap planning.

A starting list of Phase-C ideas is in `CLAUDE.md` ("Phase C — your own ideas"). Pick from that list, propose your own, or both. For every Phase-C idea you pursue:

- Write the goal in `reports/<exp_id>.md` before launching, so it's not just a fishing expedition.
- Estimate VRAM and CPU footprint; identify a co-scheduling partner.
- If the idea is high-risk-high-reward (e.g., training a small Transformer voice tracker, fine-tuning MusicGen with LoRA, integrating diffusion-based pitch refinement), prefer it over yet another hyperparameter sweep on already-tuned components.
- Combine ideas when sensible. MERT features + a Transformer voice tracker is a more interesting experiment than either alone.

When you run out of ideas, read more papers (arXiv ISMIR/ICASSP 2024–2025, Papers With Code AMT/melody-extraction leaderboards). Running out of ideas is not a stop condition.

## Step 5 — keep going until interrupted

The only stop condition is the human interrupting. Loop:

1. Refer to `task_description_v2.md` and `results_v1_evaluation.md` to keep priorities anchored.
2. Pick the next idea from `PLAN.md`.
3. Plan co-scheduling.
4. Execute.
5. Report.
6. Commit and push.
7. Repeat.

Do not pause to ask permission. Do not stop and wait. The human may be asleep.
