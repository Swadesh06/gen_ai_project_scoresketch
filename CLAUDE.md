# CLAUDE.md — HumScribe Phase G Autonomous Build & Improve

You are an autonomous coding/research agent. You build HumScribe Phase G on top of the Phase A→E + partial Phase F codebase, get every spec gate to pass, then keep improving until the human stops you. Optimize relentlessly for the best end-to-end results, both quantitative (MV2H is the headline metric with per-axis sub-scores, plus stage-wise gates) and qualitative (rendered SVG scores, audio→score fidelity).

You do not pause to ask permission. You do not stop and wait. The human may be asleep. The loop runs until manually interrupted.

## Current project state — Phase E mostly complete, partial Phase F, Phase G starting

**This is a continuation, not a fresh start.** Three prior sessions built Phase A through partial Phase F. Before doing anything new, read the prior artifacts (see "What to read first" below). Don't re-run completed gates, don't re-implement working modules.

### What's done (do NOT redo)

**Phase A** — all four spec gates pass.

**Phase B+1** — production defaults: hybrid voicing (`pesto_crepevoicing`), voice tracking with adaptive_pj, hand-tuned thresholds.

**Phase B+2 (v3.4 spec)** — five of six items shipped:
- Rendering polish (tuplet caps, integer tempo, KrumhanslSchmuckler key, render at tpb=12)
- YourMT3+ as default for Romantic-detected piano
- MusicGen-Melody-Large as Stage 7 arrangement
- Voicing exit hysteresis (tried, +0.5pp vs target +5pp, **discarded**)
- MedleyDB pseudo-labeling replaced with MTG-QBH (capped at heuristic, **discarded**)

**Phase D** (autonomous) — three landed:
- **B76 Transformer voice tracker**: 94.47% mean held-out accuracy on Romantic ASAP
- **B77 MusicGen LoRA infrastructure**: r=32, 8.57 GB VRAM peak, validated training loop
- **B79 per-voice DP**: +1.66pp Chopin Berceuse, auto-routes only on Chopin-style

**Phase E (v3 task spec)** — strict scorecard 2/8 pass, but 7 production improvements shipped:
- **Item 1 MV2H metric**: Java wrapper, IO converters, eval scripts. **The new headline metric.** Per-axis sub-scores expose where headroom is.
- **Item 8 MAESTRO regen**: re-rendered at `render_tpb=8` — integer tempo, key signature, **zero unreadable tuplets**. The visual headline win.
- **F-1 octave sanity corrector** for `beat_this`: +0.0101 mean MV2H, +0.088 on Bach BWV 856. **Default-on.**
- **tpb=24 → tpb=12 production default**: +0.011 mean MV2H, confirmed independently by sweep and ME-14.
- **F-2e formant offset detector**: +0.0508 on Vocadito offset20 F1, +0.0028 MV2H. **Opt-in flag** (worst-case per-piece regression violates strict criterion).
- **C5b r=64 LoRA adapter** on JSB Chorales: test loss 0.983 (vs B77 distill 1.39). **Production default.**
- **HuggingFace MusicGen backend** behind env switch (Windows cross-platform).

Strict v3 failures, all documented honestly: Item 2 MIR-ST500 stack (close), Item 3 DDSP ensemble (definitively negative on violin), Item 6 MV2H sweep (real ceiling at +0.022), Item 7 ensemble members (ME-1 through ME-12 all discarded).

### Current headline numbers

- ASAP 9-piece mean MV2H (real beats, full production) = **0.5492** (up from 0.5277 pre-Phase-E baseline)
- Bach BWV 856 MV2H = 0.5588 (up from 0.4589, **+0.100** from F-1 octave sanity alone)
- Vocadito A1 noff F1 = **0.666** (canonical mir_eval), IAA ceiling 0.740 — do not chase above 0.74
- Vocadito offset20 F1 = 0.439 baseline, **0.494 with F-2e enabled** (opt-in)
- Vocadito MV2H = 0.5079
- MAESTRO instrument multi-pitch F1 = 0.984 (saturated)

### MV2H sub-axis breakdown — the KEY INSIGHT for Phase G

Per-axis on ASAP:
- multi-pitch: 0.962 (**saturated** — YourMT3+ is at ceiling)
- value (duration): 0.989 (**saturated** — DTW absorbs near-coincidence)
- voice: 0.704 (**headroom** — B76 outputs aren't plumbed into MV2H text)
- meter: 0.103 (**huge headroom** — tatum grid isn't emitted)
- harmony: 0.000 (**untapped** — no chord lines emitted)

Per-axis on MAESTRO:
- multi-pitch: 0.928
- voice: **0.463** (even more headroom than ASAP)
- meter: 0.138
- value: 0.764
- harmony: 0.000

**The remaining wins are in the metric emission, not in transcription.** This is the framing for Phase G.

### Qualitative branch state

**Instrument pipeline**: Bach Fugue piano is sight-readable. MAESTRO chamber demo is publishable. Romantic piano (Beethoven, Schumann, Chopin) is approximate but recognizable. Liszt is structurally broken (DP can't represent extreme rubato; acknowledged out-of-scope).

**Humming pipeline**: melody is reliably captured; rhythmic detail is approximate. For a course-project demo, both branches are strong. Neither is replace-a-human accurate.

### What Phase G adds

Full spec in `task_descriptions/task_description_v4.md`. 17 work items in 4 stages:

- **Stage 1 (7 items, CPU-only, one-day)**: emitter fixes + published post-processing tricks
  - G-1 voice ID plumbing, G-2 meter grid markers, G-3 F-1b second-signal octave detector
  - G-4 same-pitch merging (CREPE Notes 2023), G-5 median pitch smoothing (pYIN), G-6 silent trimming
  - G-7 pre-recorded demo hums
- **Stage 2 (5 items)**: new signal + diagnostics
  - G-8 round-trip self-consistency metric (Cohen 2020)
  - G-9 confidence-aware per-note output (free signal from existing models)
  - G-10 bar-level consistency diagnostic
  - G-11 render_tpb auto-detect, G-12 ME-14 system-level ensemble selection
- **Stage 3 (3 items, GPU)**: bigger lifts
  - G-13 Lakh MIDI LoRA training (replaces C5b's data-bound 315 pairs)
  - G-14 multi-take averaging UX
  - G-15 DDSP solo_flute2 retest (less vibrato-sensitive than violin)
- **Stage 4 (2 items)**: close-out
  - G-16 C5b subjective listening test (human), G-17 Docker actual-build verification (user)

After Stage 1 alone the project is in publishable course-project state.

## What to read first (in this order, before anything else)

1. `gen_ai_project_scoresketch/CLAUDE.md` — this file. Full read, including the parallelization section and the OOM protocol.
2. `gen_ai_project_scoresketch/PLAN.md` — live plan from prior sessions. Full read; append Phase G section.
3. `gen_ai_project_scoresketch/reports/PHASE_E_SESSION_SUMMARY.md` and `reports/PHASE_E_v3_STRICT_SCORECARD.md` — what Phase E shipped and what it strict-failed. Full read.
4. `gen_ai_project_scoresketch/reports/PHASE_D_SUMMARY.md` and `reports/PHASE_D_INTEGRATION.md` — Phase D production code. Full read.
5. `gen_ai_project_scoresketch/reports/PHASE_B_FINAL.md` and `reports/INDEX.md` — historical context. Skim.
6. **`gen_ai_project_scoresketch/reports/results_v1_evaluation.md`** — first human evaluation. Full read.
7. **`gen_ai_project_scoresketch/reports/results_v2_evaluation.md`** — second human evaluation. Full read.
8. **`gen_ai_project_scoresketch/reports/results_v3_evaluation.md`** — third human evaluation. **Required reading.** Identifies the MV2H sub-axis insight and the framing shift for Phase G.
9. **`gen_ai_project_scoresketch/task_descriptions/task_description_v4.md`** — Phase G spec. **Required reading. Refer back to it before each new experiment.**
10. `gen_ai_project_scoresketch/task_descriptions/task_description_v3.md` — Phase E spec; strict-fail items documented in v3 strict scorecard.
11. The `humscribe/` package source tree and `scripts/` — skim; re-read modules you'll touch (especially `humscribe/eval/mv2h_io.py` and `humscribe/notes/post_process.py` for Stage 1 work).

After reading, **append a new "Phase G — session start" section to `PLAN.md`** covering: (a) Stage 1 execution order with co-scheduling plan; (b) Stage 2 plan; (c) any OOM-protocol experiments anticipated; (d) Phase G ensemble member priority (carry-forward from v3 ME-14); (e) Phase H ideas you plan to pursue after Phase G.

## Environment — pack/unpack conda, secrets, the `humscribe` package

Conda env name: `humscribe`. Activate with `conda activate humscribe`.

Pack/unpack workflow:
- Source of truth = `/workspace/env-archives/humscribe.tar.gz`
- After **any** package install you must `bash /workspace/scripts/pack_env.sh humscribe`
- Caches at `/workspace/.cache/...` are persistent

Don't `pip install -e .` (rejected by conda-pack). The `humscribe.pth` file already wires the package into the env.

**Java is required for MV2H eval**: `apt install default-jre`. Repack the env after.

Secrets in `.env`: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `WANDB_API_KEY`. Load via `python-dotenv` or `export`. Never echo, never commit.

## Hardware — confirmed 16 GB RTX 2000 Ada, with explicit OOM protocol

The previous CLAUDE.md cited 32 GB Blackwell. **The actual hardware is RTX 2000 Ada with 16 GB VRAM**. Everything in Phase G fits in 16 GB, but the parallelization budget is different from what the previous CLAUDE.md described.

### VRAM budget on 16 GB

Sanity check that everything fits — runs are sequential unless explicitly co-scheduled:

| Component | VRAM | When |
|---|---|---|
| Full transcription pipeline (PESTO + CREPE + YourMT3+ + B76 + beat_this) | ~8 GB | every transcribe call |
| MusicGen-Melody 1.5B inference | ~5 GB | Stage 7 arrangement (default for live demo) |
| MusicGen-Melody-Large 3.3B inference | ~13 GB | Stage 7 arrangement (offline high-quality mode) |
| Lakh LoRA training on MusicGen 1.5B | ~10 GB | Phase G item G-13 |
| DDSP solo_flute2 inference | ~1 GB | Phase G item G-15 |
| Headroom for activations + cache | ~3 GB | always |

**You can run the full pipeline + arrangement standalone on 16 GB**. You cannot co-locate two large GPU models. You CAN always co-schedule CPU work alongside GPU work.

### OOM protocol — non-negotiable

For any experiment whose **estimated peak VRAM is ≥ 12 GB**:

1. **Dry-run first.** Launch with `nvidia-smi --query-gpu=memory.used --format=csv -l 1 > logs/vram_<exp_id>.log` for the first 60 s. Record actual peak.
2. **If peak < 14 GB**: continue at planned batch size.
3. **If peak ≥ 14 GB**: halve the batch size, retry the dry-run. Repeat until peak < 14 GB.
4. **If batch size = 1 still OOMs**: log the incident to `reports/_OOM_INCIDENTS.md` with:
   - experiment ID
   - model name + size
   - peak VRAM observed
   - what was tried (batch sizes attempted)
   - whether the experiment was abandoned or completed via workaround
5. **Stop the experiment after recording the OOM**. Don't try further workarounds — let the user know via the report.

Items that need the dry-run protocol in Phase G:
- **G-13 Lakh MIDI LoRA training** (estimated ~10 GB on MusicGen-Melody 1.5B). Stay on 1.5B; do NOT attempt Large.
- **MusicGen-Melody-Large inference** for high-quality arrangement (~13 GB; fits standalone but can't co-locate)

Everything else in Phase G is < 6 GB peak; no dry-run needed.

**Never full-fine-tune MusicGen.** Always LoRA. Full fine-tuning of any MusicGen variant on 16 GB will OOM at batch=1.

### Compute targets

- **VRAM**: ≤ 14 GB (87% of 16 GB), 12.5% buffer for fragmentation
- **GPU compute**: ≥ 80% during active workload
- **CPU**: ≥ 50% of cores busy whenever GPU is busy. **Phase G is mostly CPU-bound (Stage 1 + most of Stage 2); saturate cores aggressively.**

## Parallelization rules — non-negotiable

The work splits into two largely independent resource classes. Plan every experiment by which class it lives in, then schedule it alongside experiments from the other class.

**GPU-bound work**:
- YourMT3+ / ByteDance / PESTO / CREPE / beat_this inference
- B76 training, BiLSTM training, LoRA training
- MusicGen / DDSP inference
- Any sweep agent that loads a model directly (cache features first to make it CPU-only)

**CPU-bound work** (most of Phase G):
- **MV2H evaluation** (G-1, G-2, G-3, G-8 evaluation portion)
- **Cemgil-Kappen DP rhythm quantization**
- **Most Stage 1 items** (G-1 through G-7 are all CPU)
- **G-8 round-trip self-consistency** (FluidSynth + MFCC distance)
- **G-9 confidence aggregation, G-10 bar-level diagnostic, G-12 ME-14**
- music21 score construction, Verovio SVG rendering, mir_eval scoring
- Dataset prep (Lakh MIDI download + FluidSynth render for G-13 data)
- Hyperparameter sweep orchestration

### Concrete co-scheduling patterns (use these)

1. **GPU train + CPU eval**: while a model runs on GPU, eval of the *previous* run's outputs runs on CPU in parallel. Never serialize.
2. **GPU inference batch + CPU rendering pipeline**: pipe note events to CPU rendering workers as they emerge.
3. **CPU-only stages co-running with GPU stages**: Stage 1 (all CPU) runs alongside Stage 3 (GPU). Default state: 1 GPU job + multiple CPU jobs + monitor.
4. **Sweep parallelism**: cache features once, then ~6 CPU sweep agents on cores.
5. **Dataset prep in background**: Lakh MIDI download + render runs in `prep-lakh` tmux while everything else happens.
6. **Always-on CPU worker pool**: maintain a `cpu-worker` tmux running a watcher loop that picks up un-evaluated outputs and computes MV2H + per-axis sub-scores.

Process for every new experiment: write down in `PLAN.md` which resource class, peak VRAM/CPU, and the co-scheduling partner. If you can't name a partner, find one. **Default state of the box: ≥ 1 GPU job + ≥ 1 CPU job + monitor — always.**

### Hard guardrails

1. **Dry-run any new experiment** for 30-60 s before launching the full run. For ≥ 12 GB estimated peak, the OOM protocol is mandatory.
2. **Never run a single experiment idle on the GPU.** Fill spare capacity with CPU work.
3. **Continuous monitoring**: `monitor` tmux running `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` + htop. If GPU < 50% for > 5 min with jobs queued, launch more CPU work.
4. **VRAM safety**: if `nvidia-smi` reports > 14 GB during a non-MusicGen-Large workload, kill the lowest-priority job.
5. **Process isolation**: co-scheduled jobs run in separate Python processes (separate tmux), not threads.

## tmux — every long-running thing goes in a session

Disconnections must not kill work. Anything > 30 s → tmux.

Naming:
- `monitor` — nvidia-smi + htop
- `cpu-worker` — always-on CPU eval watcher
- `train-<exp_id>` — training runs
- `infer-<exp_id>` — inference runs
- `eval-<gate>` — evaluation scripts
- `sweep-<name>-<n>` — sweep agents (numbered)
- `prep-<dataset>` — dataset prep (`prep-lakh`, `prep-jsb`, etc.)
- `roundtrip-<exp_id>` — G-8 round-trip evaluation runs
- `emitter-<g-id>` — Phase G Stage 1 emitter fix work
- `dryrun-<exp_id>` — OOM protocol dry-runs

```bash
tmux new -d -s <name> 'cd /workspace/swadesh/gen_ai_project_scoresketch && conda activate humscribe && <cmd> 2>&1 | tee logs/<name>.log'
tmux capture-pane -t <name> -p -S -200
```

Always redirect to `logs/`. Never lose scrollback.

## Logging — WandB mandatory for every run

Project: `humscribe-v3.2` (don't fork). One run per experiment.

Required:
- **Config**: full hyperparameters, git SHA, dataset name+version, model+checkpoint hash, mode, seed
- **Scalars**: train/val loss, lr, epoch, step
- **Metrics**: **always include all five MV2H sub-scores** (`mv2h_multi_pitch`, `mv2h_voice`, `mv2h_meter`, `mv2h_value`, `mv2h_harmony`) plus `mv2h_mean`. Plus stage-wise metrics (`beat_f_measure`, `note_F1`, `COnP_F1`, `offset20_F1`).
- **G-8 round-trip distance** when available
- **Qualitative artifacts**: rendered SVGs (before AND after for rendering changes), piano rolls, pitch contours, attention maps
- **VRAM peak** from the dry-run log
- **Tags**: `phase-g`, `metric-mv2h`, `metric-roundtrip`, `emitter-fix` (for G-1/G-2), `confidence-output` (for G-9), `gate`, `sweep`, `ablation`, `baseline`

Init:
```python
import wandb
wandb.init(project="humscribe-v3.2", name=exp_id, config=cfg, tags=tags,
           dir="logs/wandb", reinit=False)
```

Local logs alongside WandB: `logs/<exp_id>.log` + `reports/<exp_id>.md`.

## Checkpointing — every training run

- Save every N steps (~5-10 min cadence)
- `checkpoints/<exp_id>/step_<N>.pt`, plus `last.pt` symlink
- **Keep latest 4 checkpoints**, rolling deque
- Save: model + optimizer + scheduler + RNG states + step + epoch + best metric + config
- `--resume <path>` and `--resume latest` must produce bit-identical continuation
- `best.pt` separately, not counted in the 4-deep limit

## Reports — one per experiment

Write `reports/<exp_id>.md` for every experiment. Mandatory sections:

```
# <exp_id> — <one-line summary>

## Goal
Map to work item in task_description_v4.md.

## Procedure
Code paths, hyperparameters, dataset, seeds, hardware, VRAM peak.
**What was co-scheduled with this job.**
For OOM-protocol items: dry-run results, batch sizes attempted, final config.

## Results
Tables and numbers. WandB URL. Artifact paths.
**All five MV2H sub-scores AND mean MV2H** for every change.
For rendering changes: before/after SVG paths.
ASAP numbers cited with beat source.

## Interpretation
Why it worked / didn't. What this rules in/out.
For Stage 1 emitter fixes: which sub-axis moved and by how much.
For Stage 2 signals: correlation with MV2H |r|.

## Next
Next experiment, linked.
```

No "I've successfully…", no superlatives. State facts.

Maintain `reports/INDEX.md` chronologically.

## Improvement playbook

### Phase A through Phase E — done. Do not redo.

### Phase G — execute task_description_v4.md

Items 1-17 in 4 stages. Stage 1 (CPU-only, 7 items) is the highest-priority half-day. Stages 2-4 follow.

**Co-schedule aggressively**. Stage 1 + Stage 3 can run simultaneously (Stage 1 fills CPU while Stage 3 runs on GPU). Stage 2 fills CPU after Stage 1 lands.

### Phase H — your own ideas after Phase G

Once Phase G is stable, ideate. Read literature (arXiv ISMIR/ICASSP 2024-2026, Papers With Code). Hardware is 16 GB; everything in Phase H must fit, and the OOM protocol applies.

Residual gaps to consider for Phase H:
- The 27pp ASAP score-beats vs real-beats gap — F-1 closed +0.0101 mean; further beat-correction post-processing might find more
- The 22pp Vocadito offset20 gap — F-2e closed +0.0508 on offset20 but +0.0028 on MV2H; the next move is a learned offset detector that uses confidence-weighted aggregation
- The harmony sub-axis at 0.000 — a chord recognition module (ME-6 candidate) would lift this from zero
- MusicGen LoRA generalization — after G-13 Lakh training, try a smaller fine-tune on a specific style

Don't try: more BiLSTM voicing on 40-clip Vocadito, more MIR-ST500 pretraining (wrong domain), fine-tuning beat_this on ASAP (already trained on it), full fine-tune of MusicGen.

## Git — commit every meaningful step

Account: **Swadesh06**. SSH already configured. Always SSH URLs.

Commits:
- Session start: `chore: phase g session start, read v3 evaluation and v4 task description`
- Per work item: `g-N(<scope>): <one-line result>` with metric in body
- Per experiment: `exp(<exp_id>): <one-line idea>`
- Per report: `report(<exp_id>): <result one-liner>`
- OOM incidents: `incident(<exp_id>): OOM at <vram_peak> on <model>`
- PLAN.md updates

Push to `origin` after every commit.

Never commit: `.env`, secrets, `checkpoints/`, `logs/`. Always commit: `reports/`, `task_descriptions/`, code.

## Coding rules

- Short names. Loop iterators 2-3 letters.
- No emojis in print/log.
- No narration comments.
- No inline imports.
- Exhaustive switch handling for `Literal[...]` / enums.
- No bare `except:`.
- Type hints on every public function.
- Seeds set at every entry point, logged.

## The improvement loop

After Phase G stabilizes, loop forever:

1. **Refer back to `task_description_v4.md` and `results_v3_evaluation.md`** before picking the next item. Priorities don't drift.
2. Pick next idea from `PLAN.md` "Phase H ideas".
3. Estimate VRAM. If ≥ 12 GB, apply OOM protocol.
4. **Plan co-scheduling**: name a CPU partner if launching GPU work. Default ≥ 1 GPU + ≥ 1 CPU + monitor.
5. Launch in tmux, log to WandB with `phase-h` tag.
6. While running, design the next experiment.
7. When finished, write report, update INDEX, commit, push.
8. **Always check rendered SVG and per-axis MV2H sub-scores**, not just mean MV2H.
9. If improved: integrate. If not: discard with reasoning.
10. **Repack env** if packages changed.
11. Be aggressive with the next idea, but constrained by the OOM protocol.
12. Goto 1.

Stop conditions: human interrupts. Running out of ideas is not a stop condition.

## Output discipline (talking to the human)

Direct. Plain. No superlatives. No "groundbreaking". State result + number + evidence path. Bullet lists over prose for facts. One paragraph of interpretation max.

That's it. Read PLAN, three evaluations, v4 task description, prior reports. Plan with parallelization in mind. Apply OOM protocol where needed. Execute. Improve. Don't stop.
