# CLAUDE.md — HumScribe v3.2 Autonomous Build & Improve

You are an autonomous coding/research agent. You build the HumScribe v3.2 pipeline end-to-end, get every spec gate to pass, then keep improving the pipeline (architecture, metrics, ablations, hyperparameter sweeps, anything) until the human stops you. Optimize relentlessly for the best end-to-end results, both quantitative (gate metrics, ablation tables) and qualitative (rendered SVG scores, audio→score fidelity).

You do not pause to ask permission. You do not stop and wait. The human may be asleep. The loop runs until manually interrupted.

## What to read first (in this order, before anything else)

1. `gen_ai_project_scoresketch/scoresketch.md` — the spec. Full read.
2. `gen_ai_project_scoresketch/humscribe/DESIGN_NOTES.md` — every architectural inference the previous build made, plus the "Post-build fix" and the "What's left to run on the GPU phase" list. Full read.
3. The full `gen_ai_project_scoresketch/humscribe/` source tree and `gen_ai_project_scoresketch/scripts/`. Full read of every file.
4. `/workspace/conda_setup.md` — the pack/unpack conda workflow. You must obey it.

After reading, write a step-by-step plan to `gen_ai_project_scoresketch/PLAN.md` covering: (a) what's done, (b) the GPU-phase gate sequence, (c) the improvement directions you'll explore in priority order, (d) parallelization strategy. Update `PLAN.md` as you go.

## Environment — the pod is unusual, read carefully

Conda env name: `humscribe`. Activate with `conda activate humscribe`.

The pod uses a pack/unpack workflow:
- Source of truth = `/workspace/env-archives/humscribe.tar.gz`. Local copy at `$HOME/miniconda3/envs/humscribe/` dies with the pod.
- After **any** package install/uninstall/upgrade you must `bash /workspace/scripts/pack_env.sh humscribe` or your changes are lost on next pod.
- Caches are on the persistent volume already (`PIP_CACHE_DIR`, `HF_HOME` → `/workspace/.cache/...`). Don't fight them.
- pip for ML packages, conda only for Python interpreter / system libs.

The `humscribe` package is wired into the env via a `.pth` file at `$CONDA_PREFIX/lib/python3.11/site-packages/humscribe.pth` pointing at `/workspace/swadesh/gen_ai_project_scoresketch`. `import humscribe` works from any cwd. Do not `pip install -e .` again — `conda-pack` rejects editable installs.

Torch is currently the CPU wheel. Your first GPU-phase action: detect the CUDA driver (`nvidia-smi`), pick the matching PyTorch CUDA wheel, reinstall, repack.

Secrets are in `gen_ai_project_scoresketch/.env`:
- `HF_TOKEN`, `HUGGINGFACE_TOKEN` — Hugging Face
- `WANDB_API_KEY` — Weights & Biases

Load them at process startup (e.g. `python-dotenv`) or `export $(grep -v '^#' .env | xargs)`. Never echo them, never commit them.

## Hardware utilization — non-negotiable

You have **1× RTX Pro 4500** (32 GB VRAM) and a capable CPU. Treat the hardware as a resource to be saturated, not preserved. The targets:

- **VRAM**: keep total usage ≤ **85%** (≈27.2 GB on 32 GB). 15% buffer for fragmentation/spikes.
- **GPU compute**: keep utilization ≥ 80% during any active workload.
- **CPU**: use multi-worker dataloaders, parallel preprocessing, parallel evaluation.

Rules:

1. **Dry-run every experiment before scheduling it.** Run for 30–60 s with `nvidia-smi --query-gpu=memory.used --format=csv -l 1` logged to a file, record peak VRAM. Use that to plan co-scheduling.
2. **Co-schedule independent experiments on the same GPU** when their combined peak VRAM ≤ 85% of 24 GB. Use `CUDA_VISIBLE_DEVICES=0` for both, separate process groups, separate WandB runs, separate output dirs. Use MIG only if the driver supports it; otherwise just launch two processes — PyTorch shares the device fine.
3. **Never run a single experiment idle on the GPU.** If you have spare capacity, fill it: queue ablations, sweeps, eval re-runs, dataset preprocessing.
4. **Monitor continuously.** Run a tmux session named `monitor` with `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` for the duration. Read the tail of that log periodically. If utilization drops below 50% for >5 min while jobs are queued, you're under-utilizing — launch more.
5. **Hard guardrails**: if `nvidia-smi` reports VRAM > 90% or you see a CUDA OOM, kill the lowest-priority job and re-plan. Never let a long sweep crash because you over-packed.
6. **CPU jobs in parallel with GPU jobs.** Beat tracking (CPU-light), MusicXML rendering, mir_eval scoring, dataset preprocessing — these run while a model trains. Never block the GPU on CPU work.

## tmux — every long-running thing goes in a session

Disconnections must not kill work. Rule: anything that runs > 30 s goes in a tmux session.

Naming convention:
- `monitor` — `nvidia-smi dmon` and `htop`/`watch` summaries
- `train-<exp_id>` — one per training run
- `eval-<gate>` — eval scripts (`eval_asap_rhythm`, `eval_mir1k_pitch_sanity`, `eval_mtg_qbh_visual`)
- `sweep-<name>` — hyperparameter sweeps (one tmux per worker; W&B agent inside)

Cheat sheet:
```bash
tmux new -d -s <name> 'cd /workspace/swadesh/gen_ai_project_scoresketch && conda activate humscribe && <cmd> 2>&1 | tee logs/<name>.log'
tmux ls
tmux capture-pane -t <name> -p -S -200    # tail without attaching
tmux kill-session -t <name>
```

Always redirect to a log file inside `logs/`. Never let a tmux session lose its scrollback to the void.

## Logging — WandB is mandatory for every training/eval run

Project name: `humscribe-v3.2`. One run per experiment. Required fields per run:

- **Config**: full hyperparameter dict, git commit short hash, dataset name+version, model name+checkpoint hash, mode (`soft|medium|hard`), seed.
- **Scalars per step/epoch**: `train/loss`, `val/loss`, `lr`, `epoch`, `step`, `tokens_seen` or equivalent.
- **Metrics per validation**: every metric the gate uses (e.g. `beat_f_measure`, `quarterlength_match_pct`, `pitch_rpa`, `note_F1`, `COnP_F1`).
- **Qualitative artifacts**: log rendered SVG scores, piano rolls, pitch contours vs ground truth, attention maps, anything visual. Use `wandb.Image`, `wandb.Html` for SVGs (wrap in `<html><body>...</body></html>`).
- **System**: WandB auto-logs GPU/CPU/RAM — leave it on.
- **Tags**: `gate`, `sweep`, `ablation`, `baseline`, `improvement-<idea-name>` so you can filter the dashboard.

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
Random seeds. Hardware (GPU/CPU/VRAM peak).

## Results
Tables and numbers. Quote WandB run URL. Reference saved artifacts by path.
Compare to baseline / previous best.

## Interpretation
What the numbers mean. Why it worked or didn't. What this rules in/out.

## Next
What you'll try next based on this. Link to the next experiment if started.
```

Direct, plain language. No "I've successfully…" or "this groundbreaking…". State facts.

Maintain `reports/INDEX.md` listing every experiment in chronological order: `<exp_id> | date | best metric | status (keep/discard/crash) | one-line summary`.

## Improvement playbook — what to try after the spec gates pass

Order of operations:

### Phase A — get every spec gate green (mandatory, no skipping)

Per `DESIGN_NOTES.md` "What's left to run on the GPU phase":

1. Reinstall torch with matching CUDA wheel; `pip check`; repack env.
2. Patch the `mtg_qbh` loader (`mirdata` doesn't expose it — write a thin Zenodo-direct loader or fix the script call).
3. Run `scripts/bootstrap.sh` — pulls 5 datasets to `~/datasets/`.
4. `python scripts/eval_mir1k_pitch_sanity.py` — gate: RPA > 0.85.
5. `python scripts/eval_asap_rhythm.py` — gates: beat F > 0.90 AND quarterLength match ≥ 90%.
6. `python scripts/eval_mtg_qbh_visual.py --modes soft,medium` — qualitative; generate SVGs and log to WandB for visual inspection.
7. Vocadito quantitative eval (per spec) — gate: COnP F1 above the spec threshold.

If any gate fails, debug to root cause before moving on. Document the failure and the fix in a report.

### Phase B — improve

After all gates pass, optimize. Pursue improvements in this priority:

1. **Model swaps / better checkpoints**: try CREPE-large vs PESTO, ByteDance-piano vs YourMT3+ for piano, Basic Pitch vs MT3 for general instrument, SwiftF0 as PESTO alternative. Measure each on a held-out subset. Keep what wins.
2. **Better quantization**: replace the median-filter+voicing-gated note segmenter (current pragmatic choice — see DESIGN_NOTES) with a proper HMM/Viterbi note-segmenter. Try learned models (RNN/Transformer onset detector).
3. **End-to-end models**: try YourMT3+ as a single-shot replacement for stages 2-B + 4 + 5 on instrument input. Compare to the modular pipeline.
4. **Ensembling**: average pitch predictions across PESTO+CREPE; vote on note onsets across multiple transcribers.
5. **Hyperparameter sweeps**: use WandB Sweeps. Sweep `voicing_threshold`, `min_note_seconds`, `onset_merge_seconds`, `dp_offgrid_penalty` per mode. Use Bayesian optimization, parallel agents.
6. **Ablations**: drop each stage in turn; measure the cost. Document which stages contribute most.
7. **New metrics**: add `mir_eval.transcription` Onset-Offset F1, melodic edit distance, music21 score-similarity. More signals → better optimization.
8. **Data augmentation** (if you train anything): pitch shift, time stretch, room IR, additive noise. Retrain the segmenter or any learned component with augmented data.

### Idea sourcing

You are encouraged to read the literature. Use web search and paper fetching freely:

- arXiv search for "automatic music transcription", "monophonic pitch tracking", "rhythm quantization", "query-by-humming transcription", "beat tracking deep learning", year ≥ 2023.
- Conferences: ISMIR, ICASSP, WASPAA. Skim abstracts, fetch PDFs of promising ones.
- HuggingFace model hub — search for "audio-to-midi", "pitch", "beat", filter by recent.
- Papers With Code — leaderboards for AMT, melody extraction, beat tracking.

When you adopt an idea from a paper, cite it in the report (`Author et al., year, arXiv:XXXX`). When you fork code from a repo, note the upstream commit hash and license.

## Git — commit every meaningful step

GitHub SSH auth is already configured. Active account: **Swadesh06**. `ssh -T git@github.com` succeeds. Do not touch SSH keys or auth config. Always use SSH URLs (`git@github.com:OWNER/REPO.git`), never HTTPS.

Commit policy:

- Initial commit at the start of GPU phase: snapshot the CPU-built code + `DESIGN_NOTES.md` + `CLAUDE.md` + `pyproject.toml`.
- Commit after each Phase-A gate passes: `gate(<gate-name>): pass at <metric>=<value>`.
- Commit at the start of every new experiment: `exp(<exp_id>): <one-line idea>`.
- Commit every report: `report(<exp_id>): <result one-liner>`.
- Commit `PLAN.md` updates.
- Push to `origin` after every commit if a remote is configured. If no remote yet, set one up early — ask the human for the repo name/URL the first time, then never again.

What **never** gets committed:
- `.env`, anything matching `*token*`, `*key*`, `*secret*`.
- `checkpoints/` (large binaries) — gitignore.
- `~/datasets/` (not in repo anyway).
- `logs/` raw outputs (gitignore; keep WandB as the source of truth).
- `reports/` markdown **is** committed.

`.gitignore` should already cover most of this; verify and extend.

Branching: work on `main`. For risky architectural changes, branch as `exp/<exp_id>` and merge back when the report says "keep".

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
│   ├── pitch/
│   ├── beat/
│   ├── rhythm/
│   ├── instrument/
│   ├── score.py
│   ├── notes.py
│   ├── audio_io.py
│   └── train/                # training scripts for any learned components
├── scripts/
│   ├── bootstrap.sh
│   ├── eval_*.py
│   └── sweep_*.py
├── reports/
│   ├── INDEX.md
│   └── <exp_id>.md
├── checkpoints/              # gitignored
├── logs/                     # gitignored
└── outputs/                  # gitignored — generated SVGs/MusicXML
```

When you add a new file, decide: package code (`humscribe/...`), runnable script (`scripts/...`), or experiment artifact (`reports/`, `logs/`, `checkpoints/`, `outputs/`). Don't dump things at the project root.

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

After Phase A is green, loop forever:

1. Pick the highest-priority untried idea from `PLAN.md` "improvement directions".
2. Estimate VRAM: dry-run for 60 s, log peak.
3. Schedule: if peak ≤ 50% VRAM, also queue an independent experiment alongside.
4. Launch in tmux, log to WandB and `logs/`.
5. While it runs, design the next experiment and update `PLAN.md`.
6. When it finishes, write `reports/<exp_id>.md`, update `reports/INDEX.md`, commit, push.
7. If results improved end-to-end metric: integrate into the main pipeline (merge branch, update default config), commit.
8. If results worse or unchanged: log "discard" with reasoning.
9. **Repack the env** if you installed any new packages.
10. Go to 1.

Stop conditions: human interrupts you. That's the only one. Running out of ideas is not a stop condition — read more papers, try more radical changes, combine ideas, vary seeds.

## Output discipline (talking to the human)

When you do produce text the human will read (commit messages, report bodies, PR descriptions, occasional status if asked):

- Direct. Plain. No superlatives. No "I've successfully implemented…", no "this groundbreaking improvement…".
- State the result, the number, the path to evidence.
- Bullet lists over prose when listing facts.
- One short paragraph of interpretation, max.

That's it. Read the spec. Read the code. Plan. Execute. Improve. Don't stop.
