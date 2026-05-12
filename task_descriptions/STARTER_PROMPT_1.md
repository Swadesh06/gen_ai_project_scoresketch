# Starter Prompt — HumScribe v3.2 Autonomous Run

You are taking over an in-progress project. Your operating instructions live in `gen_ai_project_scoresketch/CLAUDE.md` — read that file in full before doing anything else, then follow it. It covers: the env, hardware utilization rules, tmux discipline, WandB logging, checkpointing, reports, git, repo layout, the improvement playbook, and the never-stop loop.

## Where things stand right now (CPU-phase complete)

- Project root: `/workspace/swadesh/gen_ai_project_scoresketch/`
- Spec: `scoresketch.md` (HumScribe v3.2 — humming/instrument audio → MusicXML/SVG via PESTO/CREPE pitch, ByteDance/Basic-Pitch transcription, beat_this beat tracking, Cemgil–Kappen DP rhythm quantization, three new validation datasets: ASAP, MTG-QBH, MIR-1K).
- Codebase: `humscribe/` package fully scaffolded (config, pipeline, pitch/, beat/, rhythm/, instrument/, score, audio_io). Three eval scripts in `scripts/` per spec §D.
- Conda env `humscribe` (Python 3.11, **CPU torch wheel**) is built and packed at `/workspace/env-archives/humscribe.tar.gz`. The package is wired into the env via a `.pth` file pointing at the project root, so `import humscribe` works from any cwd. Do NOT `pip install -e .` again — `conda-pack` rejects editable installs. See `humscribe/DESIGN_NOTES.md` "Post-build fix" for the full story.
- Hardware now: **1× RTX Pro 4500 (24 GB VRAM)** + capable CPU.
- Secrets in `gen_ai_project_scoresketch/.env`: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `WANDB_API_KEY`. Use them; never commit them.
- GitHub: SSH auth already configured for account `Swadesh06`. `ssh -T git@github.com` succeeds. Use SSH URLs only.

## Your job, in order

1. Read `CLAUDE.md` end-to-end.
2. Read `scoresketch.md` end-to-end.
3. Read `humscribe/DESIGN_NOTES.md` end-to-end (especially the "Post-build fix" and "What's left to run on the GPU phase" sections — those are your immediate to-do list).
4. Skim every file in `humscribe/` and `scripts/`.
5. Write `PLAN.md` at the project root: covering (a) Phase A gate sequence, (b) parallelization plan, (c) prioritized improvement ideas for Phase B. Update it as you work.

### Phase 0 — fix the known-broken items before anything else

These issues are documented in `DESIGN_NOTES.md` and **block** every gate. Fix them in order, verify each, before moving past Phase 0. No experiments, no eval runs, no improvement loop until Phase 0 is fully green.

#### 0.1 — Verify environment integrity

```bash
conda activate humscribe
which python                                         # expect /home/swadesh/miniconda3/envs/humscribe/bin/python
python -c "import humscribe; print(humscribe.__file__)"   # must work from any cwd
ls /workspace/env-archives/humscribe.tar.gz          # tarball must exist
cat $CONDA_PREFIX/lib/python3.11/site-packages/humscribe.pth   # must point at /workspace/swadesh/gen_ai_project_scoresketch
```

If any of those fail, stop and diagnose. The CPU-build state is the contract you're starting from — confirm it before changing anything.

#### 0.2 — Initialize WandB and test it works

```bash
set -a; source gen_ai_project_scoresketch/.env; set +a
wandb login --relogin "$WANDB_API_KEY"
python -c "import wandb; r = wandb.init(project='humscribe-v3.2', name='_smoke', mode='online'); wandb.log({'smoke': 1.0}); r.finish()"
```

If the smoke run doesn't show up in your WandB project, fix it before continuing — every later step depends on it.

#### 0.3 — Initialize the git repo and remote

```bash
cd /workspace/swadesh/gen_ai_project_scoresketch
git init -b main 2>/dev/null || true
# .gitignore: .env, checkpoints/, logs/, outputs/, ~/datasets, *.tar.gz, __pycache__/, *.pyc, .ipynb_checkpoints/
# (write/extend .gitignore now if it doesn't already cover these)
git add .gitignore CLAUDE.md PLAN.md scoresketch.md pyproject.toml humscribe/ scripts/
git status                                           # confirm .env is NOT staged
git commit -m "chore: initial GPU-phase snapshot of CPU-built code"
```

If no GitHub remote exists yet, ask the human **once** for the repo URL (SSH form, e.g. `git@github.com:Swadesh06/<repo>.git`), set it, push. After this single question, never ask the human anything again.

```bash
git remote add origin <ssh-url-from-human>
git push -u origin main
```

#### 0.4 — Reinstall torch with the matching CUDA wheel, then repack

```bash
nvidia-smi                                           # note the CUDA driver version (top right)
conda activate humscribe
pip uninstall -y torch torchaudio torchvision
# Pick the CUDA index matching the driver. RTX Pro 4500 with a recent driver -> cu124 typically:
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
pip check
conda deactivate
bash /workspace/scripts/pack_env.sh humscribe       # critical — without repack, the install dies with the pod
ls -la /workspace/env-archives/humscribe.tar.gz     # confirm new mtime
```

If the cu124 wheel doesn't match your driver (`pip check` complains, or the assert fires), try cu121 / cu118 — pick the highest CUDA the driver supports. Repack again after the swap.

Commit: `git commit -am "fix(env): swap torch to CUDA wheel matching driver"` and push.

#### 0.5 — Patch the `mtg_qbh` loader

`mirdata` 1.0.0 (and surrounding versions) does **not** expose `mtg_qbh` — `mirdata.initialize("mtg_qbh", ...)` raises. Both `scripts/bootstrap.sh` and `scripts/eval_mtg_qbh_visual.py` call it.

Verify the failure first:

```bash
python -c "import mirdata; print('mtg_qbh' in mirdata.list_datasets())"   # expect False
```

Fix: write `humscribe/datasets/mtg_qbh.py` — a thin Zenodo-direct loader with the same interface the eval script expects (`d.load_tracks() -> dict[track_id, Track]` where each `Track` exposes `.audio_path`). Source: Zenodo record `1290712`. Then update both `scripts/bootstrap.sh` (replace the `mirdata.initialize("mtg_qbh", ...)` block with a call to your loader's `download()`) and `scripts/eval_mtg_qbh_visual.py` (swap the `mirdata.initialize` line for `from humscribe.datasets.mtg_qbh import MTGQBH; d = MTGQBH(data_home=mtg_dir)`).

Verify:

```bash
python -c "from humscribe.datasets.mtg_qbh import MTGQBH; d = MTGQBH(); d.download(); ts = d.load_tracks(); print(len(ts), 'tracks; first:', next(iter(ts.values())).audio_path)"
```

Commit: `git commit -am "feat(datasets): add Zenodo-direct mtg_qbh loader; mirdata doesn't expose it"` and push. Append a note to `DESIGN_NOTES.md` "Post-build fix" recording what you did.

#### 0.6 — Run the dataset bootstrap

In a tmux session named `bootstrap`:

```bash
tmux new -d -s bootstrap 'cd /workspace/swadesh/gen_ai_project_scoresketch && conda activate humscribe && bash scripts/bootstrap.sh 2>&1 | tee logs/bootstrap.log'
```

Periodically `tmux capture-pane -t bootstrap -p -S -100` to monitor. When done, verify each dataset:

```bash
ls -la ~/datasets/{vocadito,maestro,mtg_qbh,asap,mir1k}
du -sh ~/datasets/*
```

All five must be present. If any failed, fix and re-run. Commit a brief `reports/phase0_bootstrap.md` documenting per-dataset sizes and any issues.

### Phase 0 done = ALL of the following true simultaneously

- `humscribe` imports from any cwd, env tarball is fresh
- WandB online smoke run visible in dashboard
- git repo initialized, `.env` unstaged, pushed to GitHub
- `torch.cuda.is_available()` is True, env repacked
- `mtg_qbh` loader works, both scripts updated
- All 5 datasets on disk at `~/datasets/`
- Every step above committed and pushed

Only after all six bullets are true do you move to Phase A (the spec gates per `CLAUDE.md` "Phase A — get every spec gate green"). After Phase A, Phase B (the improvement loop). Run forever. Saturate the GPU. Co-schedule. Read papers. Commit and push after every meaningful step.

Do not stop to ask whether to continue. Do not summarize and pause after Phase 0 — keep going straight into Phase A. The human may be asleep. The only stop condition is manual interruption.

Begin by reading `CLAUDE.md`.
