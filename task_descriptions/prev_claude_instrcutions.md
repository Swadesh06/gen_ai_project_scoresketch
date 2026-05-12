# CLAUDE.md — HumScribe v3.4 Autonomous Build & Improve

You are an autonomous coding/research agent. You build the HumScribe v3.4 pipeline end-to-end, get every spec gate to pass, then keep improving the pipeline (architecture, metrics, ablations, hyperparameter sweeps, anything) until the human stops you. Optimize relentlessly for the best end-to-end results, both quantitative (gate metrics, ablation tables) and qualitative (rendered SVG scores, audio→score fidelity).

You do not pause to ask permission. You do not stop and wait. The human may be asleep. The loop runs until manually interrupted.

## Current project state — Phase B+1 stable, GPU-phase ongoing

**This is a continuation, not a fresh start.** The previous session built and shipped Phase A + Phase B+1. Before doing anything new, **read the existing PLAN.md, the existing reports/INDEX.md, and the human-supplied evaluation and next-steps docs** (see "What to read first" below). The starter prompt for this session also tells you to do this — don't re-run completed gates, don't re-implement working modules.

What's already done (do **not** redo):
- Phase A: all four spec gates pass (`gate_mir1k`, `gate_asap`, `gate_vocadito_soft_A1`, `gate_mtg_qbh_visual`).
- Phase B+1 production defaults are live: TPB=24, hybrid voicing (`pesto_crepevoicing`, vt=0.75, psw=19), voice tracking with adaptive_pj on, B36b + B49 wins integrated.
- 30+ Phase B experiments documented under `reports/` with per-experiment markdown + JSON.
- Headline numbers (these are the current baselines you build on top of, not targets to re-achieve):
  - MIR-1K mean RPA = 0.988
  - ASAP BWV 846 beat F = 0.915, Stage-5 snap = 0.847
  - ASAP mean Stage-5 across 5 Bach Fugues = 0.856
  - ASAP mean Stage-5 across 5 mixed (1 Bach + 4 Romantic) = 0.590 (B49 with adaptive_pj)
  - Vocadito A1 soft F1 = 0.665, A2 = 0.630
  - **Vocadito IAA ceiling = 0.740** (B51) — pipeline is 7.5–11pp below human agreement; do not chase above 0.74
  - Vocadito offset20 F1 = 0.439 vs IAA offset20 = 0.642 — **22pp gap is the biggest unfixed weakness on the humming side**
  - **ASAP upstream loss = 18.8pp = 100% from ByteDance** (B58 oracle test) — 0pp from beat tracking, 0pp from DP. This is the biggest unfixed weakness on the instrument side.
  - MAESTRO instrument F1 (sanity) = 0.984
- 50+ commits, 60+ WandB runs at https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2

What needs to happen next (in priority order, all detailed in `task_descriptions/task_description_v2.md`):

1. **Demo-critical rendering polish** — tempo display rounding, tuplet denominator capping, render-time TPB=12, KrumhanslSchmuckler key estimation, side-by-side SVG diff in experiment template. The evaluation report flagged that current SVGs contain 24-lets and 48-lets that no human would read; metrics didn't catch this.
2. **YourMT3+ as a generative seq2seq transcription backend** — routed via the existing `auto_piano` heuristic for Romantic-detected pieces. ByteDance stays default for MAESTRO-style.
3. **MusicGen-Melody-Large as Stage 7 arrangement** — new Streamlit tab, melody-conditioned audio generation that uses the user's hum as the actual melody.
4. **Voicing exit-side hysteresis on humming** — addresses the B55 offset-F1 gap.
5. **Pseudo-label MedleyDB-Melody to enlarge the onset training set** — speculative, skip if pressed.
6. **Final demo polish** — gate re-runs, report figures, screen recording as demo-day fallback, README.

The exact specs, success criteria, and dependency graph are in `task_descriptions/task_description_v2.md`. **Read it before planning. Refer back to it before starting each new experiment.**

After you finish work items 1–6, the project is in good demo shape but not done. **Ideate further on your own** — read the literature, propose more advanced methods (specific Phase-C ideas to consider are listed in `task_descriptions/task_description_v2.md` and below). Be ambitious; the hardware is sized for it.

## What to read first (in this order, before anything else)

1. `gen_ai_project_scoresketch/PLAN.md` — current live plan, agent-maintained from the previous session. Full read.
2. `gen_ai_project_scoresketch/reports/PHASE_B_FINAL.md` and `reports/INDEX.md` — what's been tried, what won, what was discarded with rationale. Full read.
3. **`gen_ai_project_scoresketch/reports/results_v1_evaluation.md`** — human evaluation of the Phase B+1 results, including visual analysis of rendered SVGs and the diagnostic priorities for what to fix. **Required reading before planning any new work.**
4. **`gen_ai_project_scoresketch/task_descriptions/task_description_v2.md`** — the next-steps spec the human handed off. Six work items with concrete deliverables, pass criteria, and decision rules. **Required reading before planning any new work. Refer back to it before each new experiment so the priorities don't drift.**
5. `gen_ai_project_scoresketch/scoresketch.md` — original spec. Re-read sections relevant to anything you change.
6. `gen_ai_project_scoresketch/humscribe/DESIGN_NOTES.md` — historical architectural decisions and the post-build fix list.
7. The full `gen_ai_project_scoresketch/humscribe/` source tree and `gen_ai_project_scoresketch/scripts/`. Skim — re-read modules you're about to touch.
8. `/workspace/conda_setup.md` — the pack/unpack conda workflow. You must obey it.

After reading, **update** `gen_ai_project_scoresketch/PLAN.md` (don't replace; append a new Phase B+2 section) covering: (a) which of the six work items in `task_description_v2.md` you'll execute first, (b) how you'll co-schedule them on CPU + GPU for maximum hardware utilization, (c) the success criteria you're targeting per work item, (d) the Phase-C ideas you're saving for after the six items are done. Update `PLAN.md` after every meaningful step.

## Environment — the pod is unusual, read carefully

Conda env name: `humscribe`. Activate with `conda activate humscribe`.

The pod uses a pack/unpack workflow:
- Source of truth = `/workspace/env-archives/humscribe.tar.gz`. Local copy at `$HOME/miniconda3/envs/humscribe/` dies with the pod.
- After **any** package install/uninstall/upgrade you must `bash /workspace/scripts/pack_env.sh humscribe` or your changes are lost on next pod.
- Caches are on the persistent volume already (`PIP_CACHE_DIR`, `HF_HOME` → `/workspace/.cache/...`). Don't fight them.
- pip for ML packages, conda only for Python interpreter / system libs.

The `humscribe` package is wired into the env via a `.pth` file at `$CONDA_PREFIX/lib/python3.11/site-packages/humscribe.pth` pointing at `/workspace/swadesh/gen_ai_project_scoresketch`. `import humscribe` works from any cwd. Do not `pip install -e .` again — `conda-pack` rejects editable installs.

Torch is GPU-built and working (Phase A cleared). If you need to upgrade for compatibility (e.g., audiocraft requires `torch>=2.4` for MusicGen support), reinstall and **repack the env** after.

Secrets are in `gen_ai_project_scoresketch/.env`:
- `HF_TOKEN`, `HUGGINGFACE_TOKEN` — Hugging Face
- `WANDB_API_KEY` — Weights & Biases

Load them at process startup (e.g. `python-dotenv`) or `export $(grep -v '^#' .env | xargs)`. Never echo them, never commit them.

## Hardware utilization — non-negotiable, parallelization is the priority

You have **1× RTX Pro 4500 Blackwell** (32 GB VRAM) and a capable CPU. Treat the hardware as a resource to be saturated, not preserved. Single-job execution wastes the box; you should be running multiple workloads in parallel essentially all the time.

Targets:

- **VRAM**: keep total usage ≤ **85%** (≈27.2 GB on 32 GB). 15% buffer for fragmentation/spikes.
- **GPU compute**: keep utilization ≥ 80% during any active workload.
- **CPU**: keep at least 50% of cores busy whenever the GPU is busy. The CPU has independent capacity that should not sit idle.

### Parallelization rules (read carefully — this is critical)

The work splits into two largely independent resource classes. **Plan every experiment by which class it lives in, then schedule it alongside experiments from the other class.**

**GPU-bound work** (saturates VRAM and GPU compute):
- Pretrained-model inference: ByteDance, YourMT3+, MusicGen, beat_this batches, CREPE-full, PESTO
- Any training run (BiLSTM onset detector, etc.)
- Any sweep agent that runs a model

**CPU-bound work** (uses CPU cores, negligible GPU):
- Cemgil-Kappen DP rhythm quantization (pure NumPy)
- music21 score construction + `makeNotation`
- Verovio SVG rendering
- mir_eval scoring (note-level F1, COnPOff, beat F-measure, RPA)
- Audio I/O, resampling, RMS normalization
- Dataset preprocessing (FluidSynth rendering of MAESTRO MIDI, MTG-QBH unzipping, MIR-1K audio extraction)
- Hyperparameter sweep *orchestration* (the agent process, not the worker)
- Side-by-side SVG diffing, report figure generation
- Voice tracking (greedy assigner is pure NumPy)
- Voicing hysteresis evaluation on cached pitch traces

**Concrete co-scheduling patterns you must use:**

1. **GPU train/inference + CPU eval**: while a model runs on GPU, evaluation of the *previous* run's outputs (note-F1, beat-F, snap%) runs on CPU in parallel. Never serialize "GPU run finishes → CPU eval starts → next GPU run". Always: "GPU run N starts → as soon as GPU run N's outputs land on disk, kick off CPU eval N in parallel with GPU run N+1".
2. **GPU inference batch + CPU rendering pipeline**: when running the full pipeline on a dataset (e.g. all 40 Vocadito clips through PESTO + CREPE), have a CPU worker pool consuming the note events as they emerge and producing MIDI / MusicXML / SVG. Don't wait for all 40 to be transcribed before starting to render.
3. **Two GPU experiments co-located**: if `nvidia-smi` shows a single workload using <50% VRAM (which most non-MusicGen runs will), launch a second independent experiment on the same GPU. Examples:
   - PESTO inference (~1 GB) + ByteDance batch (~3 GB) on different audio files → both fit easily, both should run simultaneously.
   - A WandB sweep agent running CREPE-tiny voicing experiments + a separate Verovio rendering job + a third process running mir_eval on cached predictions → three workloads, one box, GPU and CPU both saturated.
   - When MusicGen-Large is loaded (~13 GB), don't co-schedule another large GPU model — but absolutely do run CPU work in parallel (renderings, evaluation, plot generation, dataset preprocessing).
4. **Sweep parallelism**: when running a hyperparameter sweep, launch multiple WandB agents in separate tmux sessions. The optimal count is `floor(VRAM_total * 0.85 / VRAM_per_run)` for GPU-bound sweeps, and `floor(num_cpu_cores * 0.75)` for CPU-bound sweeps.
5. **Dataset preprocessing in background**: any time you're about to run an experiment on a new dataset, kick off the dataset prep in a tmux session in parallel with whatever else is running. Don't wait until the GPU is free to start unzipping audio.
6. **Always-on CPU worker pool**: maintain a tmux session `cpu-worker` that runs a watcher loop checking for un-evaluated outputs and computing metrics. So new outputs from GPU runs get scored automatically without a manual launch step.

**Process for every new experiment**: before launching, write down (in `PLAN.md` or in the experiment's report draft) which resource class it lives in, what its peak VRAM / CPU usage is, and what it can be co-scheduled with. If you can't name a co-scheduling partner, find one. The default state of the box should be "≥ 1 GPU job + ≥ 1 CPU job + monitor" — never "1 GPU job, everything else idle".

### Hard guardrails

1. **Dry-run every new experiment** for 30–60 s with `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1` logged to a file, record peak VRAM and steady-state utilization. Use that to plan co-scheduling.
2. **Never run a single experiment idle on the GPU.** If you have spare capacity, fill it: queue ablations, sweeps, eval re-runs, dataset preprocessing, rendering. The default question after launching a job is "what else can run alongside this".
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
- `eval-<gate>` — eval scripts (`eval_asap_rhythm`, `eval_mir1k_pitch_sanity`, `eval_mtg_qbh_visual`, `gate_vocadito_conp`)
- `sweep-<name>-<n>` — hyperparameter sweep workers (one tmux per agent, numbered)
- `prep-<dataset>` — dataset preprocessing jobs

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
- **Metrics per validation**: every metric the gate uses (e.g. `beat_f_measure`, `quarterlength_match_pct`, `pitch_rpa`, `note_F1`, `COnP_F1`, `offset20_F1`).
- **Qualitative artifacts**: log rendered SVG scores, piano rolls, pitch contours vs ground truth, attention maps, anything visual. Use `wandb.Image`, `wandb.Html` for SVGs (wrap in `<html><body>...</body></html>`). **For any experiment that affects rendered output, log the SVG before AND after** so the visual change is reviewable.
- **System**: WandB auto-logs GPU/CPU/RAM — leave it on.
- **Tags**: `gate`, `sweep`, `ablation`, `baseline`, `improvement-<idea-name>`, `phase-c`, `rendering`, `genai-yourmt3`, `genai-musicgen` so you can filter the dashboard.

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

Write `reports/<exp_id>.md` for every experiment (run, sweep, ablation). Sections, in this order, no fluff:

```
# <exp_id> — <one-line summary>

## Goal
What you tried to verify or improve. One paragraph.

## Procedure
Exact steps. Code paths touched. Hyperparameters changed. Dataset(s).
Random seeds. Hardware (GPU/CPU/VRAM peak). What was co-scheduled with this job.

## Results
Tables and numbers. Quote WandB run URL. Reference saved artifacts by path.
Compare to baseline / previous best. For any experiment that changes notation,
include before/after rendered SVG paths.

## Interpretation
What the numbers mean. Why it worked or didn't. What this rules in/out.

## Next
What you'll try next based on this. Link to the next experiment if started.
```

Direct, plain language. No "I've successfully…" or "this groundbreaking…". State facts.

Maintain `reports/INDEX.md` listing every experiment in chronological order: `<exp_id> | date | best metric | status (keep/discard/crash) | one-line summary`.

## Improvement playbook

Order of operations:

### Phase A — done. Do not redo.

All four spec gates passed in the previous session. Do not re-run them as a starting point. If you change something fundamental that could regress them, re-run the affected gates as part of your verification (item 6 of the next-steps spec).

### Phase B+1 — done. Do not redo.

30+ Phase B experiments documented in `reports/`. The kept improvements are integrated into production defaults. Do not revisit decisions that were already made unless you have a specific new reason — read the relevant `exp_B*.md` first.

### Phase B+2 — execute the six work items in `task_descriptions/task_description_v2.md`

The six items, briefly (full specs in the task description):

1. **Demo-critical rendering polish** — five sub-items (BPM rounding, tuplet denominator capping, render TPB=12, key estimation, SVG diff). Pass criterion: rendered Bach BWV 854 SVG free of 24/48-lets, no regression on snap metric.
2. **YourMT3+ as a generative seq2seq backend** — wire as third option behind `auto_piano` for Romantic-detected pieces. Pass criteria target Beethoven snap ≥ 0.92 and mixed-mean ≥ 0.74 (B58 oracle ceilings). Decision rule for default-flip in the spec.
3. **MusicGen-Melody-Large as Stage 7 arrangement** — new Streamlit tab, melody-conditioned. Pass criteria: end-to-end hum→arrange works, all 6 style presets produce coherent output, peak VRAM < 20 GB.
4. **Voicing exit-side hysteresis on humming** — sweep `vt_exit ∈ {0.25, 0.35, 0.45, 0.55, 0.65}` with `vt_enter=0.75` fixed. Pass criterion: Vocadito A1 offset20 F1 ≥ 0.50 with no no-offset regression. Decision rule in spec.
5. **MedleyDB-Melody pseudo-labeling for enlarged onset training** — speculative. Pass criterion: Vocadito A1 no-offset F1 ≥ 0.69. Skip if pressed.
6. **Final demo polish** — gate re-runs, report figures, screen recording, README updates.

The six items are largely independent (see dependency graph in the spec). **Co-schedule aggressively**: items 1, 2, 3, 4 can all run partial work in parallel because they touch different parts of the codebase and use different resources (item 1 is mostly CPU work on rendering, item 2 is GPU inference on a new model, item 3 is GPU inference on a different new model, item 4 is CPU-bound sweep on cached pitch traces).

### Phase C — your own ideas, after the six items are done

When work items 1–6 are stable and merged, ideate on your own. **Be aggressive and ambitious — the hardware (32 GB Blackwell, always-on, no caps) is sized for it.** You're not constrained to "small consumer GPU" thinking anymore.

A starting list of Phase-C ideas worth considering. Pick from this list, propose your own, or do both:

- **MERT or MusicFM features** as input to a learned segmenter. The agent's HuBERT BiLSTM (B52) underperformed at 0.592, but MERT is music-trained and might do better. Speculative.
- **Train a small Transformer voice tracker** for Romantic ASAP. The greedy + adaptive-pj voice tracker hits a ceiling on dense chordal textures (Liszt at 0.078). A Transformer over note sequences predicting voice membership could close ~30pp on hard pieces.
- **Diffusion-based pitch refinement**. Recent work uses denoising diffusion on f0 contours and note sequences. Less proven; would let you say "we tried diffusion".
- **Score completion / continuation with Music Transformer or Anticipatory Music Transformer**. Transcribe a hummed phrase, feed to Music Transformer as a prompt, generate a continuation. Bolt-on demo, fully generative.
- **Soft-IAA scoring as the headline metric**. Average F1 across A1 and A2 ground truths per clip. Lower variance, more honest, was on the agent's TODO list (B51 follow-up).
- **YourMT3+ replacing the entire stages 2-A + 4 + 5 stack on instrument input**. The current modular pipeline + new YourMT3+ backend is item 2; the more aggressive variant is to test YourMT3+ as a single-shot end-to-end transcriber and compare against the modular pipeline.
- **AudioLDM2 or MAGNeT as MusicGen alternatives** — different generative audio architectures, useful for ablation.
- **Larger Vocadito-style training set via data augmentation**: combine Vocadito + MTG-QBH (with hand-aligned ground truth for 5 clips, ~30 min in MuseScore) + synthesized humming via a TTS model conditioned on melody. Three datasets pseudo-merged → 200+ training clips.
- **Beat-conditioned DP**: the agent's B58 found beat tracking is essentially perfect on ASAP. But on real humming, beat tracking is the weakest link below 60 BPM. Investigate a learned beat tracker fine-tuned on slow vocal content (the `beat_this` model is fine-tuneable).
- **End-to-end fine-tuning of MusicGen-Melody on a small "melody → arrangement" pair set**. The `audiocraft/musicgen-dreamboothing` codebase supports LoRA fine-tuning at ~10–16 GB VRAM. Could specialize the arranger to a particular style or to follow melody more strictly.
- **Cross-attention from transcription to arrangement**: a tighter coupling between Stage 6 output and Stage 7 — feed the MIDI directly to MusicGen rather than relying on chromagram conditioning.

Don't run all of these. Pick the highest-EV ones based on what's open after items 1–6 land. Use the `reports/<exp_id>.md` template for each.

## Idea sourcing

You are encouraged to read the literature. Use web search and paper fetching freely:

- arXiv search for "automatic music transcription", "monophonic pitch tracking", "rhythm quantization", "query-by-humming transcription", "beat tracking deep learning", "melody-conditioned music generation", year ≥ 2024.
- Conferences: ISMIR, ICASSP, WASPAA. Skim abstracts, fetch PDFs of promising ones.
- HuggingFace model hub — search for "audio-to-midi", "pitch", "beat", "music generation", filter by recent.
- Papers With Code — leaderboards for AMT, melody extraction, beat tracking, music generation.

When you adopt an idea from a paper, cite it in the report (`Author et al., year, arXiv:XXXX`). When you fork code from a repo, note the upstream commit hash and license.

## Git — commit every meaningful step

GitHub SSH auth is already configured. Active account: **Swadesh06**. `ssh -T git@github.com` succeeds. Do not touch SSH keys or auth config. Always use SSH URLs (`git@github.com:OWNER/REPO.git`), never HTTPS.

Commit policy:

- Initial commit when this session starts: `chore: phase b+2 session start, read evaluation and v2 task description`.
- Commit after each work item from `task_description_v2.md` lands: `item-N(<scope>): <one-line result>` with the metric in the body.
- Commit at the start of every new experiment: `exp(<exp_id>): <one-line idea>`.
- Commit every report: `report(<exp_id>): <result one-liner>`.
- Commit `PLAN.md` updates.
- Push to `origin` after every commit.

What **never** gets committed:
- `.env`, anything matching `*token*`, `*key*`, `*secret*`.
- `checkpoints/` (large binaries) — gitignore.
- `~/datasets/` (not in repo anyway).
- `logs/` raw outputs (gitignore; keep WandB as the source of truth).
- `reports/` markdown **is** committed. So is `task_descriptions/`.

`.gitignore` should already cover most of this; verify and extend.

Branching: work on `main`. For risky architectural changes, branch as `exp/<exp_id>` and merge back when the report says "keep". The YourMT3+ and MusicGen integrations are good candidates for branches.

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
│   ├── beat/                 # beat_this_track
│   ├── rhythm/               # viterbi_quantize, voice_tracking, voice_hmm, voice_hdbscan
│   ├── instrument/           # piano (ByteDance), basic_pitch, yourmt3plus (NEW for item 2)
│   ├── arrange/              # NEW for item 3: musicgen.py
│   ├── score.py
│   ├── notes.py
│   ├── audio_io.py
│   ├── datasets/             # mtg_qbh + any new pseudo-labeled loaders
│   └── train/                # training scripts for any learned components
├── scripts/
│   ├── bootstrap.sh
│   ├── eval_*.py
│   ├── gate_*.py
│   ├── exp_B*.py
│   ├── compare_svgs.py       # NEW for item 1.5
│   └── sweep_*.py
├── app/
│   └── streamlit_app.py      # NEW: arrangement tab for item 3
├── reports/
│   ├── INDEX.md
│   ├── PHASE_B_FINAL.md
│   ├── PHASE_B_SUMMARY.md
│   ├── results_v1_evaluation.md  # human evaluation
│   └── <exp_id>.md
├── task_descriptions/
│   └── task_description_v2.md    # next-steps spec from human
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

After the six work items in `task_description_v2.md` are stable, loop forever:

1. **Refer back to `task_description_v2.md` and `results_v1_evaluation.md`** every time before picking the next work item. Priorities don't drift.
2. Pick the highest-priority untried idea from `PLAN.md` "Phase C ideas" (you populated this earlier).
3. Estimate VRAM: dry-run for 60 s, log peak.
4. **Plan co-scheduling**: identify a CPU-bound or independent GPU-bound experiment to run alongside. Default state of the box should always be ≥ 1 GPU + ≥ 1 CPU job + monitor. If you can't find a co-scheduling partner, at minimum kick off rendering / scoring / report-generation in parallel on CPU.
5. Launch in tmux, log to WandB and `logs/`, log before/after SVGs if rendering changes.
6. While it runs, design the next experiment, populate the next `reports/<exp_id>.md` skeleton, update `PLAN.md`.
7. When it finishes, write `reports/<exp_id>.md`, update `reports/INDEX.md`, commit, push.
8. If results improved end-to-end metric: integrate into the main pipeline (merge branch, update default config), commit. **Always check the rendered SVG visually too**, not just the metric — the agent's previous run had cases where TPB=24 won the metric and lost the visual.
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

That's it. Read the existing PLAN, the evaluation, the v2 task description, the existing reports. Plan with parallelization in mind. Execute. Improve. Don't stop.
