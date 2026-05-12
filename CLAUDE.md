# CLAUDE.md — HumScribe Phase E Autonomous Build & Improve

You are an autonomous coding/research agent. You build HumScribe Phase E on top of the existing Phase D codebase, get every spec gate to pass, then keep improving the pipeline (architecture, metrics, ablations, hyperparameter sweeps, anything) until the human stops you. Optimize relentlessly for the best end-to-end results, both quantitative (MV2H score-similarity is now the headline metric, plus stage-wise gates) and qualitative (rendered SVG scores, audio→score fidelity).

You do not pause to ask permission. You do not stop and wait. The human may be asleep. The loop runs until manually interrupted.

## Current project state — Phase D complete, Phase E starting

**This is a continuation, not a fresh start.** Two prior sessions built Phase A through Phase D. Before doing anything new, **read the prior session's artifacts** (see "What to read first" below). Don't re-run completed gates, don't re-implement working modules.

What's already done (do **not** redo):

**Phase A** — all four spec gates pass (`gate_mir1k`, `gate_asap`, `gate_vocadito_soft_A1`, `gate_mtg_qbh_visual`).

**Phase B+1** — production defaults: TPB=24, hybrid voicing (`pesto_crepevoicing`, vt=0.75, psw=19), voice tracking with adaptive_pj on. 30+ Phase B experiments documented in `reports/`.

**Phase B+2** (the v3.4 spec, `task_description_v2.md`) — most items shipped:
- Rendering polish: tempo rounded to integer, tuplet denominators capped, render at TPB=12, KrumhanslSchmuckler key estimation. **Bach BWV 854 SVG went from 22×12-lets + 9×24-lets + 6×48-lets to 7×12-lets + 1×24-let + 0×48-lets.** Three of four demo SVGs regenerated; **MAESTRO chamber file is still pre-polish — see Phase E work item 8**.
- YourMT3+ as default for Romantic-detected piano: +6.1pp on 9-piece ASAP overall, **+19.4pp on Chopin Berceuse**, +10pp on Schumann, +8.6pp on Beethoven (all on score beats).
- MusicGen-Melody-Large Stage 7 arrangement: working, all 6 style presets, peak VRAM 6.25 GB, 13 s per preset.
- Voicing exit hysteresis (item 4): tried, only +0.5pp on offset20 vs target +5pp, **discarded honestly**.
- MedleyDB pseudo-labeling (item 5): replaced with MTG-QBH pseudo-labels, capped at imitating the heuristic, discarded.

**Phase D** (autonomous, beyond the v3.4 spec) — three landed:
- **B76 Transformer voice tracker** (1.78M params, trained from scratch on 237 ASAP pieces): 94.47% mean held-out accuracy on Romantic pieces. Liszt 90.8%, Beethoven 97.4%, Schumann 94.8%, Chopin 94.9%. Integrated as `humscribe/rhythm/voice_transformer.py`, auto-routes Chopin-style pieces.
- **B77 MusicGen LoRA fine-tune**: 69% loss decay in 300 steps, r=32 LoRA on 4.72M trainable params (0.34% of base), peak VRAM 8.57 GB. **Caveat: trained on 6 distill pairs from MusicGen itself — adapter memorized those 6, doesn't generalize.** Real-pair training is Phase E work item 5.
- **B79 per-voice DP** using B76's voice predictions: +1.66pp on Chopin Berceuse, no-op on other Romantic pieces. Auto-routes only on Chopin-style.

**Current headline numbers** (your baselines, not your targets):
- MIR-1K mean RPA = 0.988 (saturated)
- ASAP Bach 5-Fugue mean snap = 0.856
- ASAP 9-piece overall MV2H = not yet measured (Phase E item 1 builds this)
- **ASAP 9-piece overall snap (score beats)** = **0.774** (with YourMT3+ default)
- **ASAP 9-piece overall snap (real beats)** = **0.506** (with target_bpm=110 fix)
- **The 27pp gap between the two is the dominant unfixed weakness on the instrument side.** Do NOT attempt to fine-tune `beat_this` on ASAP — it was already trained on ASAP plus 14 other classical-piano datasets. The fix is post-processing, not data.
- Vocadito A1 soft F1 = 0.665, A2 = 0.628, soft-IAA = 0.6466
- **Vocadito IAA ceiling = 0.740** — do not chase above it
- Vocadito offset20 F1 = 0.439 vs IAA offset20 = 0.642 — **22pp gap is the biggest unfixed weakness on the humming side**
- MAESTRO instrument F1 (sanity) = 0.984
- B76 voice tracking mean on Romantic ASAP = 94.47%
- 80+ commits, 90+ WandB runs at https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2

**What's left in the v3.4 spec to do** — one item still open: **regenerate the MAESTRO chamber demo file** (work item 8 in Phase E, trivial).

What Phase E adds (full spec in `task_descriptions/task_description_v3.md`):

1. **MV2H end-to-end score-similarity metric** — the new headline metric. Highest priority. CPU-only. Unlocks items 6 and parts of 7.
2. **MIR-ST500 pretraining stack** for the learned onset/voicing model. Targets the Vocadito gap. 10× data increase over what the agent's prior BiLSTMs had.
3. **DDSP humming→instrument→transcription experiment** (the user's idea 2). Test whether timbre-transfer + instrument pipeline + ensemble beats the direct humming pipeline.
4. **Cross-platform Docker image** for Windows/macOS/Linux deployment. Also: swap `audiocraft` for HuggingFace `transformers.models.musicgen` (cleaner deps).
5. **JSB Chorales real-pair training** for B77 MusicGen LoRA. Turns B77's pipeline into a useful artifact.
6. **MV2H-driven hyperparameter sweep** using item 1's metric as the optimization target. CPU-bound, embarrassingly parallel.
7. **Music-theory-guided ensemble members** — 14 candidates documented in v3 spec, ranked. **12 of 14 are CPU-only**, perfect for parallelization with GPU work.
8. **Regenerate MAESTRO chamber demo file** — one CLI call.

Plus a future-ideation list (text-prompt style hints, tempo-curve preservation, score-conditioned LoRA, notation editor, demo-mode pre-baked hums, video-diff outputs) documented at the end of the v3 spec. Pursue only after items 1–8 land.

After you finish items 1–8 and any ensemble members that meet the pass criteria, the project is in strong shape but never "done". **Continue ideating on your own** — read literature, propose more advanced methods, attack the remaining gaps. Be ambitious; the hardware is sized for it.

## What to read first (in this order, before anything else)

1. `gen_ai_project_scoresketch/PLAN.md` — live plan from prior sessions. Full read. Append a new Phase E section, don't replace prior content.
2. `gen_ai_project_scoresketch/reports/PHASE_D_SUMMARY.md` and `reports/PHASE_D_INTEGRATION.md` — what Phase D shipped and how it's wired. Full read.
3. `gen_ai_project_scoresketch/reports/PHASE_B_FINAL.md` and `reports/INDEX.md` — historical context. Skim.
4. **`gen_ai_project_scoresketch/reports/results_v1_evaluation.md`** — first human evaluation, identifies issues metrics didn't catch. Full read.
5. **`gen_ai_project_scoresketch/reports/results_v2_evaluation.md`** — second human evaluation, confirms Phase D gains, flags rendering gaps and the 27pp ASAP gap. **Required reading.**
6. **`gen_ai_project_scoresketch/task_descriptions/task_description_v3.md`** — Phase E spec. **Required reading. Refer back to it before each new experiment.**
7. `gen_ai_project_scoresketch/task_descriptions/task_description_v2.md` — v3.4 spec (mostly shipped, item 8 still open).
8. `gen_ai_project_scoresketch/scoresketch.md` — original spec. Re-read sections relevant to anything you change.
9. `gen_ai_project_scoresketch/humscribe/DESIGN_NOTES.md` — historical architectural decisions and the post-build fix list.
10. The full `gen_ai_project_scoresketch/humscribe/` source tree and `gen_ai_project_scoresketch/scripts/`. Skim; re-read modules you'll touch.
11. `/workspace/conda_setup.md` — the pack/unpack conda workflow. You must obey it.

After reading, **append a new "Phase E — session start" section to `gen_ai_project_scoresketch/PLAN.md`** (don't replace prior content) covering: (a) which of the eight Phase E work items you'll execute first and which order, (b) how you'll co-schedule them on CPU + GPU for maximum hardware utilization, (c) the success criteria you're targeting per work item, (d) the Phase-E ensemble members you'll attempt and their priority order, (e) any future-ideation items you plan to pick up after items 1–8. Update `PLAN.md` after every meaningful step.

## Environment — the pod is unusual, read carefully

Conda env name: `humscribe`. Activate with `conda activate humscribe`.

The pod uses a pack/unpack workflow:
- Source of truth = `/workspace/env-archives/humscribe.tar.gz`. Local copy at `$HOME/miniconda3/envs/humscribe/` dies with the pod.
- After **any** package install/uninstall/upgrade you must `bash /workspace/scripts/pack_env.sh humscribe` or your changes are lost on next pod.
- Caches are on the persistent volume already (`PIP_CACHE_DIR`, `HF_HOME` → `/workspace/.cache/...`). Don't fight them.
- pip for ML packages, conda only for Python interpreter / system libs.

The `humscribe` package is wired into the env via a `.pth` file at `$CONDA_PREFIX/lib/python3.11/site-packages/humscribe.pth` pointing at `/workspace/swadesh/gen_ai_project_scoresketch`. `import humscribe` works from any cwd. Do not `pip install -e .` again — `conda-pack` rejects editable installs.

Torch is GPU-built and working (Phase A through D cleared). If you need to upgrade for compatibility (audiocraft→transformers swap, or for MV2H Java integration), reinstall and **repack the env** after.

**Java required for Phase E item 1**: `apt install default-jre`. Repack the env after.

Secrets are in `gen_ai_project_scoresketch/.env`:
- `HF_TOKEN`, `HUGGINGFACE_TOKEN` — Hugging Face
- `WANDB_API_KEY` — Weights & Biases

Load them at process startup (e.g. `python-dotenv`) or `export $(grep -v '^#' .env | xargs)`. Never echo them, never commit them.

## Hardware utilization — non-negotiable, parallelization is the priority

You have **1× RTX Pro 4500 Blackwell** (32 GB VRAM) and a capable CPU. Treat the hardware as a resource to be saturated, not preserved. Single-job execution wastes the box; you should be running multiple workloads in parallel essentially all the time.

**Phase E is particularly parallelization-friendly:** 5 of the 8 work items are CPU-only, and 12 of the 14 ensemble members are CPU-only. You can fill the CPU with item 1 + item 6 + item 7 ensemble work while GPU runs item 2 / item 3 / item 5.

Targets:

- **VRAM**: keep total usage ≤ **85%** (≈27.2 GB on 32 GB). 15% buffer for fragmentation/spikes.
- **GPU compute**: keep utilization ≥ 80% during any active workload.
- **CPU**: keep at least 50% of cores busy whenever the GPU is busy. The CPU has independent capacity that should not sit idle. **Phase E offers more CPU-bound work than ever; saturate the cores.**

### Parallelization rules (read carefully — this is critical)

The work splits into two largely independent resource classes. **Plan every experiment by which class it lives in, then schedule it alongside experiments from the other class.**

**GPU-bound work** (saturates VRAM and GPU compute):
- Pretrained-model inference: ByteDance, YourMT3+, MusicGen, beat_this batches, CREPE-full, PESTO
- Any training run (MIR-ST500 BiLSTM, JSB Chorales LoRA)
- DDSP inference (small, ~1 GB)
- Any sweep agent that runs a model directly (cache features first if possible to make it CPU-only)

**CPU-bound work** (uses CPU cores, negligible GPU):
- **MV2H evaluation** (Phase E item 1) — the new headline metric, Java jar call ~50 ms per piece
- **Cemgil-Kappen DP rhythm quantization** (pure NumPy)
- **MV2H-driven hyperparameter sweep** (Phase E item 6) — 6+ parallel sweep agents on cached features
- **Most ensemble members from Phase E item 7** — ME-1 (pYIN), ME-2 (Goto), ME-4 (tonal-meter prior), ME-5 (phrase boundary), ME-7 (anacrusis), ME-8 (spiral key), ME-9 (line of fifths), ME-10 (meter template), ME-11 (formant onset), ME-12 (phase onset), ME-13 (voice legality), ME-14 (MV2H ensemble selection) — **all CPU**
- music21 score construction + `makeNotation`
- Verovio SVG rendering
- mir_eval scoring (note-level F1, COnPOff, beat F-measure, RPA)
- Audio I/O, resampling, RMS normalization
- Dataset preprocessing (FluidSynth rendering of JSB Chorales pairs, MIR-ST500/DALI audio download orchestration, MTG-QBH unzipping)
- **Docker build** (Phase E item 4)
- Side-by-side SVG diffing, report figure generation
- Voice tracking inference once features are cached
- Voicing hysteresis evaluation on cached pitch traces

**Concrete co-scheduling patterns you must use:**

1. **GPU train/inference + CPU eval**: while a model runs on GPU, evaluation of the *previous* run's outputs (MV2H, note-F1, beat-F, snap) runs on CPU in parallel. Never serialize "GPU run finishes → CPU eval starts → next GPU run". Always: "GPU run N starts → as soon as GPU run N's outputs land on disk, kick off CPU eval N in parallel with GPU run N+1".
2. **GPU inference batch + CPU rendering pipeline**: when running the full pipeline on a dataset (e.g. all 40 Vocadito clips through PESTO + CREPE), have a CPU worker pool consuming the note events as they emerge and producing MIDI / MusicXML / SVG. Don't wait for all 40 to be transcribed before starting to render.
3. **Two GPU experiments co-located**: if `nvidia-smi` shows a single workload using <50% VRAM (which most non-MusicGen runs will), launch a second independent experiment on the same GPU. Examples:
   - MIR-ST500 BiLSTM training (~3 GB) + DDSP timbre-transfer eval (~1 GB) + a Verovio rendering job + a third process running mir_eval/MV2H on cached predictions → three workloads, one box, GPU and CPU both saturated.
   - When MusicGen-Large is loaded (~13 GB), don't co-schedule another large GPU model — but absolutely do run CPU work in parallel (renderings, evaluation, plot generation, dataset preprocessing, MV2H sweep agents).
4. **Sweep parallelism**: when running a hyperparameter sweep, launch multiple WandB agents in separate tmux sessions. For item 6 specifically: cache PESTO/CREPE/ByteDance outputs once, then launch ~6 CPU agents in parallel. The optimal count for cached-feature sweeps is `floor(num_cpu_cores * 0.75)`.
5. **Dataset preprocessing in background**: any time you're about to run an experiment on a new dataset, kick off the dataset prep in a tmux session in parallel with whatever else is running. Don't wait until the GPU is free to start downloading MIR-ST500 or rendering JSB Chorales pairs.
6. **Always-on CPU worker pool**: maintain a tmux session `cpu-worker` that runs a watcher loop checking for un-evaluated outputs and computing metrics (MV2H, note-F1, etc.). So new outputs from GPU runs get scored automatically without a manual launch step.
7. **Ensemble members as filler work**: most of the 14 ensemble members in Phase E item 7 are CPU-only and independent. If at any point the GPU is busy and you have spare CPU, integrate the next ensemble member from the priority list (ME-9 → ME-4 → ME-11 → ME-7 → ME-10 → ME-1 → ME-14). This is ideal filler work.

**Process for every new experiment**: before launching, write down (in `PLAN.md` or in the experiment's report draft) which resource class it lives in, what its peak VRAM / CPU usage is, and what it can be co-scheduled with. If you can't name a co-scheduling partner, find one. The default state of the box should be "≥ 1 GPU job + ≥ 1 CPU job + monitor" — never "1 GPU job, everything else idle".

### Hard guardrails

1. **Dry-run every new experiment** for 30–60 s with `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1` logged to a file, record peak VRAM and steady-state utilization. Use that to plan co-scheduling.
2. **Never run a single experiment idle on the GPU.** If you have spare capacity, fill it: queue ablations, sweeps, eval re-runs, dataset preprocessing, rendering, ensemble member integrations. The default question after launching a job is "what else can run alongside this".
3. **Continuous monitoring**: keep a `monitor` tmux session running `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` and a sibling `htop`-or-equivalent CPU monitor. Read the tail periodically. If GPU utilization drops below 50% for >5 min while jobs are queued, you're under-utilizing — launch more.
4. **VRAM safety**: if `nvidia-smi` reports VRAM > 90% or you see a CUDA OOM, kill the lowest-priority job and re-plan. Never let a long sweep crash because you over-packed.
5. **Process isolation**: when co-scheduling on the same GPU, use separate Python processes (separate tmux sessions), not threads. PyTorch shares the device fine across processes; threads share the CUDA context and cause random hangs.

## tmux — every long-running thing goes in a session

Disconnections must not kill work. Rule: anything that runs > 30 s goes in a tmux session.

Naming convention:
- `monitor` — `nvidia-smi dmon` and `htop`/`watch` summaries
- `cpu-worker` — the always-on CPU evaluation/rendering watcher
- `train-<exp_id>` — one per training run
- `infer-<exp_id>` — one per inference run
- `eval-<gate>` — eval scripts (`eval_mv2h_asap`, `eval_mv2h_vocadito`, `eval_asap_rhythm`, `gate_vocadito_conp`)
- `sweep-<name>-<n>` — hyperparameter sweep workers (one tmux per agent, numbered)
- `prep-<dataset>` — dataset preprocessing jobs (`prep-mirst500`, `prep-dali`, `prep-jsb-pairs`)
- `docker-build` — the Phase E item 4 Docker build
- `ensemble-<me-id>` — Phase E item 7 ensemble member integrations (e.g. `ensemble-me9-line-of-fifths`)

Cheat sheet:
```bash
tmux new -d -s <name> 'cd /workspace/swadesh/gen_ai_project_scoresketch && conda activate humscribe && <cmd> 2>&1 | tee logs/<name>.log'
tmux ls
tmux capture-pane -t <name> -p -S -200    # tail without attaching
tmux kill-session -t <name>
```

Always redirect to a log file inside `logs/`. Never let a tmux session lose its scrollback to the void.

## Logging — WandB is mandatory for every training/eval run

Project name: `humscribe-v3.2` (keep using the existing one — don't fork). One run per experiment. Required fields per run:

- **Config**: full hyperparameter dict, git commit short hash, dataset name+version, model name+checkpoint hash, mode (`soft|medium|hard`), seed.
- **Scalars per step/epoch**: `train/loss`, `val/loss`, `lr`, `epoch`, `step`, `tokens_seen` or equivalent.
- **Metrics per validation**: every metric the gate uses (e.g. `mv2h_mean`, `mv2h_multi_pitch`, `mv2h_voice`, `mv2h_meter`, `mv2h_value`, `mv2h_harmony`, `beat_f_measure`, `quarterlength_match_pct`, `pitch_rpa`, `note_F1`, `COnP_F1`, `offset20_F1`).
- **Qualitative artifacts**: log rendered SVG scores, piano rolls, pitch contours vs ground truth, attention maps. Use `wandb.Image`, `wandb.Html` for SVGs (wrap in `<html><body>...</body></html>`). **For any experiment that affects rendered output, log the SVG before AND after** so the visual change is reviewable. The agent's prior session missed this on item 1 rendering polish and the MAESTRO file went unchanged silently.
- **System**: WandB auto-logs GPU/CPU/RAM — leave it on.
- **Tags**: `phase-e`, `gate`, `sweep`, `ablation`, `baseline`, `improvement-<idea-name>`, `metric-mv2h`, `ensemble-<me-id>`, `genai-yourmt3`, `genai-musicgen`, `genai-ddsp` so you can filter the dashboard.

Init pattern:
```python
import wandb, os
wandb.init(project="humscribe-v3.2", name=exp_id, config=cfg, tags=tags,
           dir="logs/wandb", reinit=False)
```

Local logs (in addition to WandB, never as a replacement): every run also writes a plain `logs/<exp_id>.log` and a `reports/<exp_id>.md` (see "Reports").

## Checkpointing — every training run

- Save every N steps (pick N so you checkpoint roughly every 5–10 min).
- Filename: `checkpoints/<exp_id>/step_<N>.pt`. Also save `last.pt` symlink.
- **Keep only the latest 4 checkpoints**, delete older. Implement as a rolling deque in your training loop.
- Save: model state, optimizer state, scheduler state, RNG states (torch + numpy + python), step, epoch, best metric, config.
- **Resume**: every train script accepts `--resume <path>` and `--resume latest`. Resuming must produce bit-identical continuation when seeds and data order are restored. Test resume once per new training script before launching long runs.
- Final/best checkpoint: copy to `checkpoints/<exp_id>/best.pt` (don't count toward the 4-deep limit).

## Reports — one per experiment

Write `reports/<exp_id>.md` for every experiment (run, sweep, ablation, ensemble integration). Sections, in this order, no fluff:

```
# <exp_id> — <one-line summary>

## Goal
What you tried to verify or improve. Map to the work item in task_description_v3.md.

## Procedure
Exact steps. Code paths touched. Hyperparameters changed. Dataset(s).
Random seeds. Hardware (GPU/CPU/VRAM peak). What was co-scheduled with this job.

## Results
Tables and numbers. Quote WandB run URL. Reference saved artifacts by path.
Compare to baseline / previous best. **For any experiment that changes notation,
include before/after rendered SVG paths.** Cite ASAP numbers with beat source
("score beats" or "real beats from beat_this").

## Interpretation
What the numbers mean. Why it worked or didn't. What this rules in/out.
For ensemble members: did this member have uncorrelated errors with the baseline?

## Next
What you'll try next based on this. Link to the next experiment if started.
```

Direct, plain language. No "I've successfully…" or "this groundbreaking…". State facts.

Maintain `reports/INDEX.md` listing every experiment in chronological order: `<exp_id> | date | best metric | status (keep/discard/crash) | one-line summary`.

## Improvement playbook

Order of operations:

### Phase A — done. Do not redo.

### Phase B+1 — done. Do not redo.

### Phase B+2 (the v3.4 spec) — done except for item 8.

Item 8 (regenerate MAESTRO chamber demo file) is a one-liner. Pick it up at any opportunity.

### Phase D — done. Do not redo.

### Phase E — execute the eight work items in `task_descriptions/task_description_v3.md`

The eight items, briefly:

1. **MV2H end-to-end score-similarity metric** — highest priority, CPU-only, unblocks items 6 and parts of 7. Build this FIRST so every subsequent experiment can be evaluated against the new objective.
2. **MIR-ST500 pretraining stack** — DALI v2 pretrain → MIR-ST500 fine-tune → Vocadito fine-tune for a learned onset/voicing model.
3. **DDSP humming→instrument experiment** — test the user's idea 2, with ensemble variant as the highest-EV configuration.
4. **Cross-platform Docker image** — also swap `audiocraft` for `transformers.models.musicgen` for cleaner deps.
5. **JSB Chorales real-pair LoRA training** — turn B77's pipeline into a useful artifact.
6. **MV2H-driven hyperparameter sweep** — once item 1 exists, use it as the optimization target for a large sweep over DP/voicing parameters.
7. **Music-theory-guided ensemble members** — 14 candidates ranked, target ME-9 → ME-4 → ME-11 → ME-7 → ME-10 → ME-1 → ME-14. Most CPU-only.
8. **Regenerate MAESTRO chamber demo file** — one CLI call, drop in anywhere.

The eight items are largely parallelizable. **Co-schedule aggressively**: item 1 is CPU-only and unblocks others, item 4 (Docker) is CPU-only and runs alongside any GPU work, item 7 ensemble members are CPU-only and slot in as filler. Items 2, 3, 5 are GPU-bound but each only takes a fraction of VRAM (~3 GB, ~6 GB, ~10 GB respectively), so two can co-locate.

### Phase F — your own ideas, after the eight items are done

When work items 1–8 are stable and merged (the kept ones; the discarded ones documented), ideate on your own. **Be aggressive and ambitious — the hardware (32 GB Blackwell, always-on, no caps) is sized for it.**

A starting list of Phase-F ideas is in `task_description_v3.md` "Future-ideation items" section. Plus the residual gaps you'll have identified by then:
- The 27pp ASAP score-beats vs real-beats gap (NOT solved by fine-tuning beat_this — try post-processing approaches, learned beat correctors, or stage-7-style ensemble across beat-tracking hypotheses)
- The 22pp Vocadito offset gap (a learned offset detector head with formant-band features is the right architecture to try)
- The Liszt structural ceiling (DP that allows local beat-stretching, not a fixed grid)
- The MusicGen LoRA generalization story (after item 5 lands, try LoRA fine-tunes on larger MIDI corpora like Lakh)

Don't run all of these. Pick the highest-EV ones based on what's open after items 1–8 land. Use the `reports/<exp_id>.md` template for each.

## Idea sourcing

You are encouraged to read the literature. Use web search and paper fetching freely:

- arXiv search for "automatic music transcription", "monophonic pitch tracking", "rhythm quantization", "query-by-humming transcription", "beat tracking deep learning", "melody-conditioned music generation", "music ensemble methods", year ≥ 2024.
- Conferences: ISMIR, ICASSP, WASPAA. Skim abstracts, fetch PDFs of promising ones.
- HuggingFace model hub — search for "audio-to-midi", "pitch", "beat", "music generation", filter by recent.
- Papers With Code — leaderboards for AMT, melody extraction, beat tracking, music generation.

When you adopt an idea from a paper, cite it in the report (`Author et al., year, arXiv:XXXX`). When you fork code from a repo, note the upstream commit hash and license.

## Git — commit every meaningful step

GitHub SSH auth is already configured. Active account: **Swadesh06**. `ssh -T git@github.com` succeeds. Do not touch SSH keys or auth config. Always use SSH URLs (`git@github.com:OWNER/REPO.git`), never HTTPS.

Commit policy:

- Initial commit when this session starts: `chore: phase e session start, read evaluation v2 and v3 task description`.
- Commit after each work item from `task_description_v3.md` lands: `item-N(<scope>): <one-line result>` with the metric in the body.
- Commit at the start of every new experiment: `exp(<exp_id>): <one-line idea>`.
- Commit every report: `report(<exp_id>): <result one-liner>`.
- Commit every ensemble member integration: `ensemble(<me-id>): <result one-liner>`.
- Commit `PLAN.md` updates.
- Push to `origin` after every commit.

What **never** gets committed:
- `.env`, anything matching `*token*`, `*key*`, `*secret*`.
- `checkpoints/` (large binaries) — gitignore.
- `~/datasets/` (not in repo anyway).
- `logs/` raw outputs (gitignore; keep WandB as the source of truth).
- `reports/` markdown **is** committed. So is `task_descriptions/`.

`.gitignore` should already cover most of this; verify and extend.

Branching: work on `main`. For risky architectural changes, branch as `exp/<exp_id>` and merge back when the report says "keep". The MIR-ST500 stack and the DDSP timbre-transfer experiment are good candidates for branches.

## Repo organization — keep it tidy

```
gen_ai_project_scoresketch/
├── CLAUDE.md                 # this file
├── PLAN.md                   # live, agent-maintained plan
├── pyproject.toml
├── scoresketch.md
├── .env                      # gitignored
├── .gitignore
├── humscribe/                # the package (importable everywhere)
│   ├── __init__.py
│   ├── config.py
│   ├── DESIGN_NOTES.md       # historical record + gotchas, append-only
│   ├── pipeline.py
│   ├── pitch/                # PESTO, CREPE, voicing, hmm_segment, ensemble
│   │   └── timbre_transfer/  # NEW for Phase E item 3: DDSP wrapper
│   ├── beat/                 # beat_this_track
│   ├── rhythm/               # viterbi_quantize, voice_tracking, voice_hmm, voice_transformer
│   ├── instrument/           # piano (ByteDance), basic_pitch, yourmt3plus
│   ├── arrange/              # musicgen.py (HF transformers version after item 4)
│   ├── eval/                 # NEW for Phase E item 1: mv2h.py, mv2h_io.py
│   ├── ensemble/             # NEW for Phase E item 7: each ME-N as a separate module
│   ├── score.py
│   ├── notes.py
│   ├── audio_io.py
│   ├── datasets/             # mtg_qbh + mirst500_loader + dali_loader + jsb_pairs
│   └── train/                # training scripts for any learned components
├── scripts/
│   ├── bootstrap.sh
│   ├── eval_*.py
│   ├── gate_*.py
│   ├── exp_B*.py
│   ├── exp_C*.py             # NEW Phase E experiments numbered C1+ to keep them distinct
│   ├── compare_svgs.py
│   ├── sweep_*.py
│   └── sweep_mv2h_e6.py      # NEW for item 6
├── app/
│   └── streamlit_app.py      # extended for new flags
├── reports/
│   ├── INDEX.md
│   ├── PHASE_B_FINAL.md
│   ├── PHASE_B_SUMMARY.md
│   ├── PHASE_D_SUMMARY.md
│   ├── PHASE_D_INTEGRATION.md
│   ├── results_v1_evaluation.md   # human evaluation v1
│   ├── results_v2_evaluation.md   # human evaluation v2
│   └── <exp_id>.md
├── task_descriptions/
│   ├── task_description_v2.md    # v3.4 plan, mostly shipped (item 8 still open)
│   └── task_description_v3.md    # Phase E spec, the active spec
├── Dockerfile                   # NEW for Phase E item 4
├── .dockerignore                # NEW for Phase E item 4
├── checkpoints/              # gitignored
├── logs/                     # gitignored
└── outputs/                  # gitignored — generated SVGs/MusicXML/arrangement WAVs
```

When you add a new file, decide: package code (`humscribe/...`), runnable script (`scripts/...`), Streamlit app code (`app/...`), or experiment artifact (`reports/`, `logs/`, `checkpoints/`, `outputs/`). Don't dump things at the project root.

## Coding rules

- **Short names.** No `object_features` when `obj_feats` works. Loop iterators are 2–3 letters with a one-line comment naming what they iterate.
- **No emojis or visual characters in `print` / log statements.** Plain text.
- **No narration comments.** Don't write `# loop over notes` above `for n in notes:`. Comments only for non-obvious intent or trade-offs.
- **No inline imports.** Imports at the top of each file.
- **Exhaustive switch handling** for any `Literal[...]`/enum field (`mode: soft|medium|hard`, `input_kind`, etc.). All branches handled explicitly; raise on unknown.
- **No bare `except:`.** Catch the specific exception or use `except Exception` with a logged context.
- **Type hints** on every public function signature.
- **Determinism**: set seeds at every entry point. Log them.

## The improvement loop

After the eight Phase E work items are stable, loop forever:

1. **Refer back to `task_description_v3.md` and `results_v2_evaluation.md`** every time before picking the next work item. Priorities don't drift.
2. Pick the highest-priority untried idea from `PLAN.md` "Phase F ideas" (you populated this earlier).
3. Estimate VRAM: dry-run for 60 s, log peak.
4. **Plan co-scheduling**: identify a CPU-bound or independent GPU-bound experiment to run alongside. Default state of the box should always be ≥ 1 GPU + ≥ 1 CPU job + monitor. If you can't find a co-scheduling partner, at minimum kick off rendering / scoring / MV2H eval in parallel on CPU.
5. Launch in tmux, log to WandB and `logs/`, log before/after SVGs if rendering changes.
6. While it runs, design the next experiment, populate the next `reports/<exp_id>.md` skeleton, update `PLAN.md`.
7. When it finishes, write `reports/<exp_id>.md`, update `reports/INDEX.md`, commit, push.
8. If results improved end-to-end metric: integrate into the main pipeline (merge branch, update default config), commit. **Always check the rendered SVG visually too**, not just the metric — the agent's prior runs had cases where TPB=24 won the metric and lost the visual, and where item 1 polish missed one demo file.
9. If results worse or unchanged: log "discard" with reasoning.
10. **Repack the env** if you installed any new packages.
11. Be aggressive with the next idea. The hardware is sized for it. Don't ladder up cautiously when you can try something ambitious.
12. Go to 1.

Stop conditions: human interrupts you. That's the only one. Running out of ideas is not a stop condition — read more papers, try more radical changes, combine ideas, vary seeds.

## Output discipline (talking to the human)

When you do produce text the human will read (commit messages, report bodies, PR descriptions, occasional status if asked):

- Direct. Plain. No superlatives. No "I've successfully implemented…", no "this groundbreaking improvement…".
- State the result, the number, the path to evidence.
- Bullet lists over prose when listing facts.
- One short paragraph of interpretation, max.

That's it. Read the existing PLAN, the two evaluations, the v3 task description, the existing reports. Plan with parallelization in mind. Execute. Improve. Don't stop.
