# Claude code agent project setup
_Exported on 5/3/2026 at 02:34:20 GMT+5:30 from Cursor (3.2.16)_

---

**User**

I have to set up an agent to work on my project described in @gen_ai_project_scoresketch/scoresketch.md . Since it is a Claude code agent, I need you to prepare a prompt for it. Inform it about the conda environemnt setup described in my file at /workspace/conda_setup.md , and tell it that it is a codeing agent that is to build and execute my project in @gen_ai_project_scoresketch/scoresketch.md .

After that, in the prompt, tell it that this currently is a cpu only instance and that it must first produce all the code necessary for the project. All of it, and complete the environment setup and dpeendency setups for the code, then stop and ask me to swithch this instance to a GPU instance. 

---

**Cursor**

I'll read both files first to understand the project and the conda setup, then craft a comprehensive prompt for the Claude code agent.

Now I have a thorough understanding of both the project (HumScribe v3.2 — a humming-to-MusicXML pipeline) and the conda environment workflow on this pod. Let me craft a detailed prompt for the Claude code agent.

```markdown
# Prompt for Claude Code Agent — HumScribe v3.2 Build (CPU-Only Phase)

## Who you are

You are a Claude Code agent acting as the **sole engineer** building the HumScribe v3.2 project end-to-end on a remote pod. Your job is to read the spec, write all the code, set up the environment, install every dependency, and stop at a clean handoff point so the human can switch the pod from CPU-only to GPU-enabled before you run anything that needs CUDA.

You build. You execute setup. You do **not** run anything that would require a GPU on this current CPU-only instance — that work waits for the GPU swap.

## What you're building

The full project spec is in `/workspace/swadesh/gen_ai_project_scoresketch/scoresketch.md` (HumScribe v3.2 — humming/instrument audio → MusicXML/SVG, with PESTO/CREPE pitch tracking, ByteDance/Basic-Pitch transcription, beat_this beat tracking, Cemgil-Kappen DP rhythm quantization, and three new validation datasets: ASAP, MTG-QBH, MIR-1K).

Read it in full before you write a single line. Then re-read sections C, D, and the v3 / v3.1 references it cites — the spec is an addendum that assumes v3 + v3.1 architecture is already specified elsewhere in the same doc; if anything is missing, infer the most reasonable design from the validation gates and stage descriptions, and make a note in `DESIGN_NOTES.md` of every inference you made.

Project root for code: `/workspace/swadesh/gen_ai_project_scoresketch/humscribe/` (create it). Datasets go to `~/datasets/` per the spec.

## Environment setup — read this carefully, the pod is unusual

This pod uses a **pack/unpack conda workflow** documented in `/workspace/conda_setup.md`. Read that file in full before touching conda. Key points you must respect:

- Conda envs live on local disk at `$HOME/miniconda3/envs/`, but the **source of truth is the tarball at `/workspace/env-archives/<name>.tar.gz`**. Local copies die with the pod.
- After creating or modifying an env, you **must repack** with `bash /workspace/scripts/pack_env.sh <name>` or your work is lost on the next pod.
- Caches (`PIP_CACHE_DIR`, `HF_HOME`) already point at `/workspace/.cache/...` — downloads survive pod termination, so don't fight it.
- HF token is auto-loaded from `/workspace/.secrets/hf_token`. GitHub SSH keys are already configured.
- Default to **pip** for ML packages, **conda** only for the Python interpreter and system libs (`ffmpeg`, `fluidsynth`, `fluid-soundfont-gm`).

Create a fresh env named `humscribe` with Python 3.11. Install all dependencies the spec requires (`pesto-pitch`, `torchcrepe`, `piano_transcription_inference`, `basic-pitch`, `beat_this`, `mirdata`, `mir_eval`, `music21`, `librosa`, `soundfile`, `numpy`, `scipy`, `torch` — **CPU build for now, since the pod is CPU-only**, etc.). Pin `torch` to the CPU wheel explicitly (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) — when the pod is swapped to GPU, the human will reinstall the CUDA wheel.

Repack the env (`bash /workspace/scripts/pack_env.sh humscribe`) once setup is complete, **before** you stop. This is the single most important non-code step — skipping it loses the entire install.

## What you do, in order

1. **Read the full spec** at `/workspace/swadesh/gen_ai_project_scoresketch/scoresketch.md` and the conda setup at `/workspace/conda_setup.md`. Take notes.
2. **Plan the module layout** for `humscribe/` — at minimum the spec implies: `pitch/`, `beat/`, `rhythm/`, `instrument/`, `pipeline.py`, `config.py`, plus `scripts/` for the eval scripts in spec §D and a `scripts/bootstrap.sh` per spec §C. Sketch this in a brief `DESIGN_NOTES.md` at the project root before writing code.
3. **Write all code.** Every module, every function the eval scripts in §D call (`humscribe.beat.beat_this_track.track_beats_beat_this`, `humscribe.rhythm.viterbi_quantize.viterbi_quantize_rhythm`, `humscribe.instrument.piano.transcribe_piano`, `humscribe.pitch.pesto_track.track_pitch_pesto`, `humscribe.pipeline.transcribe`, `humscribe.config.PipelineConfig`, etc.) must be implemented. The three eval scripts in spec §D.1–D.3 must exist verbatim where the spec gives them, and `scripts/bootstrap.sh` must follow spec §C.
4. **Create the conda env, install all deps (CPU torch), and repack.**
5. **Run only the steps that don't need a GPU**: env create, pip installs, `python -c "import humscribe"` smoke test, lint/syntax check (`python -m compileall humscribe/`), and the dataset bootstrap **if** you can confirm the downloads don't pull GPU-only model weights at import time. If any download triggers a CUDA call, skip it and add it to the GPU-phase TODO list.
6. **Do not** run any of the eval scripts or anything that loads a model checkpoint for inference — those need GPU.
7. **Stop and ask the human to switch the pod to a GPU instance.** Print a clear handoff message (see the end-state checklist below).

## Coding rules to follow

- **Short names.** No `object_features` when `obj_feats` works. Loop iterators are 2–3 letters with a one-line comment naming what they iterate. (User rule.)
- **No emojis or visual characters in `print` statements.** Plain text only. (User rule.)
- **No narration comments.** Don't write `# loop over notes` above `for n in notes:`. Comments only for non-obvious intent or trade-offs.
- **No inline imports.** All imports at the top of each file.
- **Exhaustive switch handling** for any union/enum types (the spec implies a `mode: "soft" | "medium" | "hard"` union — handle all branches explicitly).
- Prefer editing existing files over creating new ones once the initial scaffold exists.

## End-state checklist (must be true before you stop)

- [ ] `/workspace/swadesh/gen_ai_project_scoresketch/humscribe/` exists with all modules implemented per the spec.
- [ ] `scripts/bootstrap.sh`, `scripts/eval_asap_rhythm.py`, `scripts/eval_mir1k_pitch_sanity.py`, `scripts/eval_mtg_qbh_visual.py` exist and match spec §C, §D.1, §D.2, §D.3.
- [ ] `DESIGN_NOTES.md` documents every architectural inference you made beyond what the spec states explicitly.
- [ ] Conda env `humscribe` exists locally **and** has been repacked: `ls /workspace/env-archives/humscribe.tar.gz` succeeds.
- [ ] `conda activate humscribe && python -c "import humscribe; from humscribe.pipeline import transcribe; print('ok')"` runs without error.
- [ ] `python -m compileall /workspace/swadesh/gen_ai_project_scoresketch/humscribe/` reports zero syntax errors.
- [ ] No GPU-required step has been executed.
- [ ] You have printed a final message that says exactly:
  - what's been built,
  - what's been installed and repacked,
  - what's left to run on GPU (model downloads that pull weights into CUDA, all eval scripts, the bootstrap dataset downloads if you skipped them),
  - the literal sentence: **"Please switch this pod to a GPU instance, then resume me."**

Work carefully. Do not skip the repack step. Do not run anything that needs GPU. Stop at the handoff and wait.
```

You can copy this prompt straight into the Claude code agent's first message. A few things worth flagging before you hand it off:

- The spec is described as a "v3.2 addendum" — it says architecture and validation thresholds from v3 and v3.1 are unchanged. The agent will have to infer the v3/v3.1 module structure from the eval-script imports in §D and the stage descriptions, since those earlier docs aren't in the workspace. The prompt explicitly tells it to log those inferences in `DESIGN_NOTES.md` so you can sanity-check them.
- The CPU torch pin matters — when you swap to GPU, your first instruction to the resumed agent should be "uninstall torch, reinstall the CUDA wheel matching the GPU's compute capability, then repack the env."
- The handoff sentence is verbatim so you can grep / pattern-match for it if you want to script the pod swap.

---

**User**

What is teh environemnt that teh agent created, and is all that thsi prompt asked done? The agent worked in teh tmux session "autocode"

---

**Cursor**

I'll check what the agent did in the tmux session and verify the end-state checklist from the prompt.

Good, I can see the env exists, the project is built, and the agent finished. Let me capture the full final summary from tmux and verify the rest of the checklist (scripts, smoke test, repacked tarball timestamp, GPU-required-but-not-run claim).

Found a real issue. The env exists, torch is CPU-only, scripts and modules are all on disk — but `import humscribe` fails because the package isn't on `sys.path`. There's no `pyproject.toml` / `setup.py` and the env wasn't `pip install -e .`'d. Let me confirm and check the design notes / scroll the rest of the agent's final message before reporting.

Now I have a complete picture. Here's the verified state.

## The environment

**Conda env name:** `humscribe`
- **Location (local):** `/home/swadesh/miniconda3/envs/humscribe` (~4.1 GB)
- **Tarball (source of truth):** `/workspace/env-archives/humscribe.tar.gz` (~1.4 GB, repacked at 19:13 — i.e. *after* the install finished, so the repack succeeded)
- **Python:** 3.11.15
- **Torch:** 2.11.0+cpu, `torch.cuda.is_available() == False` — correctly the CPU wheel as instructed

## Checklist verification

| Prompt requirement | Status |
|---|---|
| `humscribe/` package with all modules per spec | Done — `__init__.py`, `config.py`, `notes.py`, `audio_io.py`, `score.py`, `pipeline.py`, plus `pitch/`, `beat/`, `rhythm/`, `instrument/` subpackages |
| `scripts/bootstrap.sh`, `eval_asap_rhythm.py`, `eval_mir1k_pitch_sanity.py`, `eval_mtg_qbh_visual.py` exist per spec §C / §D.1–3 | Done — all 4 files present at `gen_ai_project_scoresketch/scripts/` |
| `DESIGN_NOTES.md` documenting inferences | Done — thorough 182-line file (currently at `humscribe/DESIGN_NOTES.md`, not the project root the prompt asked for, but the content is solid) |
| Conda env `humscribe` exists locally | Done |
| Env repacked: `ls /workspace/env-archives/humscribe.tar.gz` succeeds | Done (1.4 GB tarball, mtime after the install) |
| `python -c "import humscribe; from humscribe.pipeline import transcribe"` runs without error | **Fails from arbitrary cwd** — the package was never installed (no `pyproject.toml`/`setup.py`, no `pip install -e .`). Import only works when the cwd is `gen_ai_project_scoresketch/` (so `humscribe/` is a sibling on the implicit path). The eval scripts in `scripts/` will fail to import `humscribe.*` unless run from the project root |
| `python -m compileall humscribe/` reports zero syntax errors | Done — clean |
| Nothing GPU-required has been executed | Confirmed — no model checkpoints downloaded, no eval scripts run |
| Final handoff message printed | Done — the agent printed the build summary in tmux |

## Other findings worth knowing

- **Bootstrap was not run.** The agent skipped `scripts/bootstrap.sh` because it identified that `mtg_qbh` is **not in `mirdata` 1.0.0** — calling `mirdata.initialize("mtg_qbh", ...)` will raise. The spec assumes it works; it doesn't. So the bootstrap would fail mid-run and partially populate `~/datasets`. This is documented in `DESIGN_NOTES.md` §"Known gotchas". You'll need to either pin the loader, write a thin Zenodo-direct loader, or comment out the MTG-QBH block before running bootstrap on the GPU pod.
- **TensorFlow noise on import** is cosmetic — `basic-pitch` pulls the GPU TF wheel and it complains about missing CUDA, but the package works on CPU.
- The `__pycache__` dirs being timestamped *before* the source files (May 2 19:10 vs 19:12) suggests the agent edited the source after its first compile; not a problem, just a sign of iteration.

## What to fix before / right after the GPU swap

1. **Make `humscribe` a proper installable package.** Easiest is a one-line `pyproject.toml` at `gen_ai_project_scoresketch/` then `pip install -e .` inside the env, then repack. Otherwise every script invocation has to be run from the project root or with `PYTHONPATH=/workspace/swadesh/gen_ai_project_scoresketch`.
2. **Move `DESIGN_NOTES.md`** up one level from `humscribe/` to `gen_ai_project_scoresketch/` if you want it where the prompt asked (minor).
3. **Fix the `mtg_qbh` loader** before running `bootstrap.sh`.
4. **Reinstall torch with the matching CUDA wheel**, then **repack the env** (`bash /workspace/scripts/pack_env.sh humscribe`) — without the repack the CUDA install dies with the pod.

Net assessment: the agent did ~95% of what was asked. The build is real, the env is real and persisted. The one substantive miss is the missing package install (`pip install -e .`) which silently breaks the smoke-test gate — easy 30-second fix when you move to GPU.

---

**User**

so the thing you mention will break the piline downstream? How do I fix this right now?

---

**Cursor**

Yes — the missing package install will break **every** script and any GPU-phase eval, because `from humscribe.pipeline import transcribe` (and the four other `humscribe.*` imports the eval scripts use) will raise `ModuleNotFoundError` unless you happen to be `cd`'d into `gen_ai_project_scoresketch/` at the moment of running.

I can fix it right now in three steps: write a minimal `pyproject.toml`, `pip install -e .` into the env, then repack. Let me do it.

Good — all subpackages have `__init__.py`, so a flat `pyproject.toml` will work cleanly.

---

**User**

I don't understand the problem

---

**Cursor**

Let me explain plainly what's wrong and why it matters.

## The problem in one sentence

The agent put the code on disk but never **installed** the `humscribe` package into the conda env, so Python can't find it from anywhere except one specific directory.

## What "installing a package" means

When you write `import humscribe` in any Python file, Python doesn't search your whole disk. It searches:
1. The current working directory (whatever folder you ran `python` from), and
2. Folders registered as "site-packages" inside the active env (i.e. where `pip install` puts things).

The `humscribe` folder is currently only in (1). It's not in (2). The env has no record that `humscribe` exists.

## Concrete demonstration of the breakage

```bash
# Works (because cwd contains a folder literally named humscribe):
cd /workspace/swadesh/gen_ai_project_scoresketch
python -c "import humscribe"   # ok

# Fails (cwd is anywhere else):
cd /tmp
python -c "import humscribe"   # ModuleNotFoundError: No module named 'humscribe'

cd /workspace/swadesh/gen_ai_project_scoresketch/scripts
python eval_asap_rhythm.py     # ModuleNotFoundError on line 5: from humscribe.beat... 
```

That last one is the real consequence: **all four scripts in `scripts/`** (`bootstrap.sh` is fine — it's bash, not Python — but the three eval scripts) start with `from humscribe.beat...`, `from humscribe.pitch...`, `from humscribe.pipeline...`. They will all crash on the import line, before they do any work, when run from inside `scripts/` (which is the natural place to run them from). That's the "downstream pipeline break" you asked about.

## Why this is true even though import worked when I tested it

When I ran the smoke test the first time, I was in `/workspace/swadesh` and it failed. Then I `cd`'d into `gen_ai_project_scoresketch/` and it worked. Same env, same Python — only the cwd changed. That's the smoking gun: it's not a real install, it's a coincidence of where you stand.

## The fix

Tell the env "register `humscribe` as a real installed package, pointing at this folder on disk". One command:

```bash
cd /workspace/swadesh/gen_ai_project_scoresketch
pip install -e .
```

The `-e` means "editable" — it doesn't copy the code, just registers a pointer. You keep editing `humscribe/*.py` in place; Python finds it from anywhere because the env now knows about it. After this, `python -c "import humscribe"` works from `/`, `/tmp`, your scripts dir, anywhere.

For `pip install -e .` to work, the project needs a `pyproject.toml` (which I already wrote a minute ago) telling pip what the package is called and where its source lives. So the two-step fix is:

1. `pyproject.toml` exists — done.
2. Run `pip install -e .` inside the env, then repack.

I started running step 2 in my last turn and you interrupted it. Want me to re-run it now? It takes about 5 seconds for the install plus ~3 minutes for the repack.

---

**User**

proceed

---

**Cursor**

Install succeeded. Now verify it works from a neutral cwd, then repack.

All five `humscribe.*` imports the eval scripts depend on now resolve from `/tmp` (a totally unrelated cwd). The TF lines are the harmless cuDNN-on-CPU warnings noted in `DESIGN_NOTES.md` — not errors.

Now repack so the install survives a pod restart.

Expected — `conda-pack` refuses editable installs because the path to source code on local disk won't exist on a fresh pod. Two clean fixes:

1. **Path-based `.pth` file** — write a one-liner that adds the project dir to `site-packages` path. Survives repack, no `pip install` needed, still picks up edits live.
2. **Real (non-editable) install** — `pip install .` copies the code into site-packages; loses the live-edit property, and you'd have to reinstall after every code change.

Option 1 is strictly better for an actively developed project on a `/workspace`-persistent path. Doing that now.

Editable install removed, `.pth` file written. The `.pth` mechanism is a built-in Python feature: any line in a `*.pth` file inside `site-packages/` gets added to `sys.path` at interpreter startup. It's just plain text, so `conda-pack` has no objection. And since `/workspace/...` is on the persistent network volume, the path remains valid on every future pod.

Verify, then repack.

Resolves cleanly. Repack now.

---

**User**

then continue

---

**Cursor**

Re-running the repack. It usually takes 2–4 minutes for an env this size, so I'm giving it ~10 minutes of headroom.

Repack completed cleanly: `100% Completed | 1min 27.0s`, tarball is `1.4G` at `/workspace/env-archives/humscribe.tar.gz`, mtime `19:34` (just now). One last sanity check that the freshly-packed env still imports from a neutral cwd, then I'll confirm the timestamps.

Done. Everything that was broken is now fixed and persisted.

## Final state

| Artifact | Path | Status |
|---|---|---|
| `pyproject.toml` | `/workspace/swadesh/gen_ai_project_scoresketch/pyproject.toml` | written 19:30 |
| `humscribe.pth` (path-injection file) | `$HOME/miniconda3/envs/humscribe/lib/python3.11/site-packages/humscribe.pth` | contents: `/workspace/swadesh/gen_ai_project_scoresketch` |
| Repacked env tarball | `/workspace/env-archives/humscribe.tar.gz` | 1.4 GB, mtime 19:34 (post-fix) |
| Smoke test from `cwd=/` | `python -c "import humscribe; from humscribe.pipeline import transcribe"` | passes |

## What I changed and why, in plain terms

1. Added `pyproject.toml` so the project is a real Python package with a name (`humscribe`).
2. Tried `pip install -e .` (editable install). It worked, but `conda-pack` refuses editable installs because they reference local-disk paths that won't survive a pod restart.
3. Replaced the editable install with a one-line `humscribe.pth` file inside `site-packages/`. Python reads `*.pth` files at startup and treats each line as an extra `sys.path` entry. Since the line points to `/workspace/...` (the persistent volume), it'll be valid on every future pod.
4. Repacked. `conda-pack` is happy with `.pth` files (they're plain text, not editable installs). The new tarball includes the `.pth`, so on the next pod, after `bash /workspace/scripts/startup.sh humscribe`, `import humscribe` will work everywhere out of the box.

## What this means for downstream

- The three Python eval scripts (`eval_asap_rhythm.py`, `eval_mir1k_pitch_sanity.py`, `eval_mtg_qbh_visual.py`) will now run from any cwd on the GPU pod. No more `ModuleNotFoundError`.
- Edits to any file under `humscribe/` are picked up immediately on the next Python invocation — no reinstall needed (same live-edit property as `pip install -e`, just achieved differently).
- The two known issues from `DESIGN_NOTES.md` are still pending and are not affected by this fix:
  1. `mtg_qbh` is missing from `mirdata` — needs a custom Zenodo loader before `bootstrap.sh` will complete.
  2. Torch is the CPU wheel — needs reinstall with the matching CUDA wheel after the GPU swap (and then another `pack_env.sh humscribe`).

---

**User**

ALos update the DESIGN file to say it was fixed. and how.

---

**Cursor**

I'll add a new section documenting the post-build fix, placed after "Known gotchas" so the chronology reads naturally.

Also worth removing the now-stale "Reinstall torch with the matching CUDA wheel; rerun `pip check`" *as the very first* step, because right after that the user should also re-run the repack — let me check the GPU TODO list and add a small note that the package-install fix is done so the agent doesn't redo it.

Updated. The `DESIGN_NOTES.md` now has:

- A new **"Post-build fix: `humscribe` package install (resolved)"** section that explains the original bug, why `pip install -e .` was tried first, why it was rejected by `conda-pack`, and the `.pth`-based replacement that was actually shipped.
- A short note at the top of the **"What's left to run on the GPU phase"** list pointing at the fix section so a future agent doesn't try to "fix" it again.

The TODO list is unchanged otherwise — the pending items (CUDA torch reinstall, `mtg_qbh` loader, bootstrap, three eval scripts, final repack) are all still real GPU-phase work.

---

**User**

Okay. Now I need you to design  a Claude.md file, for an autonomous claude code agent which will run the pipeline further, iterate over any fixes, any improvements to the architecture, autonomously. 

Provide the conda setup conetxt in it as well.

since this an experimental project, after completing the main steps described in @gen_ai_project_scoresketch/scoresketch.md , the agent can further try out improvements for the pipeline which make it perform better. It can be more complex or advanced architecture, more metrics, ablations, hyperparamters sweeps whatever. But I need the best prossible results in the end, so the agent has to optimise for best results as much as possible, both qualititatevely and quantitatively. This is critical.

The agent will have 1x RTX Pro 4500 at its disposal in the instance, along with a capable CPU. It has to ensure that at any point of time the hardware is used aggressively for maximum productivity, safely within the resource limits of the hardware (let's say 85% VRAM with 15% buffer). Experiments can use multiple workers, multiple expeirmnets can be run simultaenously, etc. The agent must continuously monitor the hardware and ensure that it is being productive with pararlelization of the work. If an experiment takes up 50% of teh GPU, while it runs, another independent expeirment can be fit into it if possible. To quickly check how much VRAM experiments use, the agent must dry run the exepriments hsortly, so that it can make informed decisions about how to scheule them on the hardware. Pipeline , paralallize, use the hardware to the maximum productivity, this is critical.

Also inform that to search for ideas it can do literature reviews or browse papers off the internet and whatever, whatever helps it to gain performance. 

For each experiment it does, it must create a markdown report for the experiment descirbing the procedure followed, the results obtained, and the interpretation of the results. It must use clear, to-the-point language, and not waste words or tokens on language that contains too much fluff or is superlative. Just be straightforward.

Inform it to commit the code ti github. GitHub SSH auth is already configured in this environment. Use normal git commands over SSH. The active GitHub account is Swadesh06, and `ssh -T git@github.com` succeeds. Do not modify SSH keys or auth setup unless explicitly asked. Tell it to Use SSH GitHub URLs, e.g. `git@github.com:OWNER/REPO.git`, not HTTPS URLs, if it needs ot clone repos.

Ask it to read @gen_ai_project_scoresketch/scoresketch.md  and the prepared codebase for context first, before proceeding to the next steps.

Ask it plan everything steo by step before jumping into it.

All experiments must be run in tmux sessions so they are uninterupted my disconnections.

For reference on writing a file with instructioosn for an autonomous agent I have provided you @program (1).md  form Andrej Karpathy's autoresearch repo.

It can find my hf_token, my wandb api key, etc in the @gen_ai_project_scoresketch/.env  if it needs them. All taining logs and statistics have to be recorded in wandb as well, apart from normal logs and stuff, include all relevant things like validations loss, any metric, training loss, epochs, steps, lr, any image the model is generated for qualitative output or validation output.

Training runs should have checkpointing frequently (only keep latest 4 checkpoints at any time) and resume functionality to be resumed from any checkpoint).

The repo should remain organized.


Fit all these instructions into a Claude.md designed to intruct the agent to autonmously build the project and keep imprving it until I stop it.

Give me the starter prompt to set off teh agent as well, where you cn inform it of the progress till now, and set it off for further.

Keep both the doc and the prompt clear, to the point and precise in language.

---

**Cursor**

I'll read the reference file from Karpathy's autoresearch repo and the `.env` to confirm what's available, then design `CLAUDE.md` and the starter prompt.

I have the reference. Now I'll write `CLAUDE.md` and the starter prompt. Both need to be precise, low-fluff, and explicit about the parallelism / VRAM-budgeting / WandB / tmux / git rules.

Now the starter prompt — short, sets context, hands off cleanly to `CLAUDE.md`.

```markdown
# Starter Prompt — HumScribe v3.2 Autonomous Run

You are taking over an in-progress project. Your operating instructions live in `gen_ai_project_scoresketch/CLAUDE.md` — read that file in full before doing anything else, then follow it. It covers: the env, hardware utilization rules, tmux discipline, WandB logging, checkpointing, reports, git, repo layout, the improvement playbook, and the never-stop loop.

## Where things stand right now (CPU-phase complete)

- Project root: `/workspace/swadesh/gen_ai_project_scoresketch/`
- Spec: `scoresketch.md` (HumScribe v3.2 — humming/instrument audio → MusicXML/SVG via PESTO/CREPE pitch, ByteDance/Basic-Pitch transcription, beat_this beat tracking, Cemgil–Kappen DP rhythm quantization, three new validation datasets: ASAP, MTG-QBH, MIR-1K).
- Codebase: `humscribe/` package fully scaffolded (config, pipeline, pitch/, beat/, rhythm/, instrument/, score, audio_io). Three eval scripts in `scripts/` per spec §D.
- Conda env `humscribe` (Python 3.11, **CPU torch wheel**) is built and packed at `/workspace/env-archives/humscribe.tar.gz`. The package is wired into the env via a `.pth` file pointing at the project root, so `import humscribe` works from any cwd. Do NOT `pip install -e .` again — `conda-pack` rejects editable installs. See `humscribe/DESIGN_NOTES.md` "Post-build fix" for the full story.
- The bootstrap script has not been run, and the `mtg_qbh` mirdata loader is broken (the package doesn't expose it — needs a thin Zenodo-direct loader). Both items are tracked in `DESIGN_NOTES.md` "What's left to run on the GPU phase".
- Hardware: this instance now has **1× RTX Pro 4500 (24 GB VRAM)** + a capable CPU.
- Secrets in `gen_ai_project_scoresketch/.env`: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `WANDB_API_KEY`. Use them; never commit them.
- GitHub: SSH auth already configured for account `Swadesh06`. `ssh -T git@github.com` succeeds. Use SSH URLs only.

## Your job, in order

1. Read `CLAUDE.md` end-to-end.
2. Read `scoresketch.md` end-to-end.
3. Read `humscribe/DESIGN_NOTES.md` end-to-end.
4. Skim every file in `humscribe/` and `scripts/`.
5. Write `PLAN.md` at the project root: covering (a) Phase A gate sequence, (b) parallelization plan, (c) prioritized improvement ideas for Phase B. Update it as you work.
6. Start the monitor tmux (`nvidia-smi dmon` to a log file).
7. Do the GPU-phase setup: detect CUDA driver, reinstall torch with the matching CUDA wheel, `pip check`, repack env.
8. Patch the `mtg_qbh` loader.
9. Run `bash scripts/bootstrap.sh` (in tmux).
10. Run the three eval scripts as gates. Fix anything that fails. Log everything to WandB. Write a report per gate.
11. Once every gate is green, enter the improvement loop in `CLAUDE.md` "The improvement loop". Run forever. Saturate the GPU. Co-schedule independent experiments when VRAM allows. Read papers when ideas run low. Commit and push after every meaningful step.

Do not stop to ask whether to continue. Do not summarize and pause. The loop runs until I interrupt you. The goal is the best end-to-end pipeline I can have when I come back — quantitatively (every gate metric maxed) and qualitatively (rendered SVG scores faithful to the input audio).

Begin by reading `CLAUDE.md`.
```

## What I built and why each piece is in there

**`CLAUDE.md`** is the persistent operating doc the agent will keep re-reading across sessions / sub-agents. It encodes the rules the human can't be there to enforce:

- **Reading order + `PLAN.md`**: forces context-load before action and creates a live artifact you can inspect.
- **Env section**: tells it not to redo the `.pth` fix and not to use `pip install -e .` (would break `conda-pack` again).
- **Hardware utilization**: hard 85% VRAM ceiling with explicit `nvidia-smi` commands, dry-run-then-schedule rule, co-scheduling rule, monitor tmux. The 24 GB number assumes RTX Pro 4500 SE — if your card is the 32 GB SKU, just edit the one number; everything else scales.
- **tmux**: naming convention so the agent (and you) can find any running job by `tmux ls`.
- **WandB**: every required field listed (config, scalars, metrics, qualitative artifacts including SVG via `wandb.Html`, system, tags) so it can't get lazy.
- **Checkpointing**: rolling-4 deque, full state including RNG, resume must be tested before long runs.
- **Reports**: a fixed template (Goal/Procedure/Results/Interpretation/Next) and an `INDEX.md`. Direct-language rule baked in.
- **Improvement playbook**: Phase A (mandatory gates) then Phase B (prioritized improvement directions), explicit instruction to read literature and cite.
- **Git**: SSH-only, `Swadesh06`, commit cadence, what never to commit, branch naming for risky experiments. I deliberately did NOT make it auto-create a remote — it's instructed to ask the human once if no remote exists, then never again.
- **Repo layout**: explicit tree so the agent knows where new things go.
- **Coding rules**: your two user rules (short names, no emojis), plus the exhaustive-switch / no-inline-imports rules that match the workspace `.cursor/rules/` files.
- **Improvement loop**: the never-stop spec, modeled after Karpathy's `LOOP FOREVER` block but adapted for multi-experiment parallel scheduling instead of a single GPU + single train.py.

**Starter prompt** is intentionally short. It only does three things: tells the agent where it is in the timeline, points at `CLAUDE.md`, and gives an ordered task list for the very first session. After session 1, every future invocation can be just `read CLAUDE.md and continue`.

A few things I want to flag explicitly:

- I assumed RTX Pro 4500 = 24 GB. If yours is 32 GB, edit `≈20.4 GB` to `≈27.2 GB` in `CLAUDE.md` § Hardware.
- The "ask the human once for the remote URL" gives the agent permission to break the never-stop rule exactly one time, for one specific question. If you'd rather pre-create the GitHub repo and put the SSH URL in `CLAUDE.md` directly, that's cleaner — let me know and I'll bake it in.
- `WANDB_API_KEY` is committed to `.env`. Make sure `.env` is in `.gitignore` *before* the agent's first `git add .` — I'd suggest doing this once now manually, or telling the agent in the starter prompt to verify it as step zero. Want me to add a `.gitignore` right now?

---

**User**

Add intructions in the prompt to have the agent fix the remaining issues you said to do on starting up the gpus instance so it does that before moving on

---

**Cursor**

The starter prompt already has those steps (CUDA torch, mtg_qbh loader, bootstrap, gates) at items 7–10, but they're terse. Let me make them explicit, ordered, and verification-gated so the agent can't skip ahead.

Here's the updated starter prompt — replace the previous one with this:

```markdown
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

These two issues are documented in `DESIGN_NOTES.md` and **block** every gate. Fix them in order, verify each, before moving past Phase 0. No experiments, no eval runs, no improvement loop until Phase 0 is fully green.

**0.1 — Verify environment integrity**

```bash
conda activate humscribe
which python                                         # expect /home/swadesh/miniconda3/envs/humscribe/bin/python
python -c "import humscribe; print(humscribe.__file__)"   # must work from any cwd
ls /workspace/env-archives/humscribe.tar.gz          # tarball must exist
cat $CONDA_PREFIX/lib/python3.11/site-packages/humscribe.pth   # must point at /workspace/swadesh/gen_ai_project_scoresketch
```

If any of those fail, stop and diagnose. The CPU-build state is the contract you're starting from — confirm it before changing anything.

**0.2 — Initialize WandB and test it works**

```bash
set -a; source gen_ai_project_scoresketch/.env; set +a
wandb login --relogin "$WANDB_API_KEY"
python -c "import wandb; r = wandb.init(project='humscribe-v3.2', name='_smoke', mode='online'); wandb.log({'smoke': 1.0}); r.finish()"
```

If the smoke run doesn't show up in your WandB project, fix it before continuing — every later step depends on it.

**0.3 — Initialize the git repo and remote**

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

**0.4 — Reinstall torch with the matching CUDA wheel, then repack**

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

**0.5 — Patch the `mtg_qbh` loader**

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

**0.6 — Run the dataset bootstrap**

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

**Phase 0 done = ALL of the following true simultaneously:**
- `humscribe` imports from any cwd, env tarball is fresh
- WandB online smoke run visible in dashboard
- git repo initialized, `.env` unstaged, pushed to GitHub
- `torch.cuda.is_available()` is True, env repacked
- `mtg_qbh` loader works, both scripts updated
- All 5 datasets on disk at `~/datasets/`
- Every step above committed and pushed

Only after all six bullets are true do you move to Phase A (the spec gates). After Phase A, Phase B (the improvement loop) per `CLAUDE.md`. Run forever. Saturate the GPU. Co-schedule. Read papers. Commit and push after every meaningful step.

Do not stop to ask whether to continue. Do not summarize and pause after Phase 0 — keep going straight into Phase A. The human may be asleep. The only stop condition is manual interruption.

Begin by reading `CLAUDE.md`.
```

## What changed and why

I split the original "items 7–10" into a **Phase 0** with six explicit, ordered, verification-gated sub-steps:

| Sub-step | What it fixes | Verification command included |
|---|---|---|
| 0.1 | Confirm CPU-build state is intact before mutating anything | `import humscribe`, `.pth` file present, tarball present |
| 0.2 | Make sure WandB works *before* it's needed (silent WandB failure is the worst-case outcome) | smoke run with online mode |
| 0.3 | Repo + remote + `.env` excluded — addresses the secret-leak risk you flagged earlier; also the one place the agent is allowed to ask the human a single question | `git status` must not show `.env` staged |
| 0.4 | The CUDA torch swap from `DESIGN_NOTES` — with explicit `pip check`, repack, and CUDA-version fallback guidance | `torch.cuda.is_available()` assert |
| 0.5 | The `mtg_qbh` loader fix — written as "build a real loader at `humscribe/datasets/mtg_qbh.py` and patch both call sites", not "comment out the broken line" | imports + load test |
| 0.6 | Run the bootstrap (which 0.5 unblocks) | all 5 dataset directories present + sized |

Each sub-step has a verification command, an explicit commit, and a push. The agent literally cannot pretend to have done it without leaving evidence in git history and on disk.

The Phase 0 done-criteria block at the bottom is a hard checklist — six bullets, all must be true — designed to prevent the agent from rationalizing skipping ahead because "Phase A is more interesting".

Two things I deliberately kept *outside* Phase 0:
- The eval gates themselves (`eval_mir1k_pitch_sanity.py`, `eval_asap_rhythm.py`, `eval_mtg_qbh_visual.py`, Vocadito) — those are Phase A in `CLAUDE.md`. Phase 0 is just "make the broken plumbing not broken"; Phase A is "run the pipeline and check it meets spec".
- The `reports/INDEX.md` initialization — that's part of Phase A's first report write per `CLAUDE.md`.

Want me to also drop a starter `.gitignore` so the agent doesn't have to invent it?

---

**User**

the complete prompt wasn't in an .md artiffact redo it

---

**Cursor**

Got it — saving the full updated starter prompt to a markdown file.

Saved to `/workspace/swadesh/gen_ai_project_scoresketch/STARTER_PROMPT.md` (172 lines). Same content as the prompt in my previous reply, now in a single markdown artifact you can copy/paste, attach, or reference by path.

Two practical notes:

- The file is at the project root next to `CLAUDE.md`. The starter prompt is a single-use artifact (only the first session needs it), so committing it to the repo is optional. If you'd rather keep it out of the GitHub repo, add `STARTER_PROMPT.md` to `.gitignore` — or just delete it after the agent's first run. If you do want it in the repo as a record of how the autonomous run was kicked off, add it to the staged set in step 0.3 (currently the agent only stages `CLAUDE.md PLAN.md scoresketch.md pyproject.toml humscribe/ scripts/`).
- The `CLAUDE.md` you have at line 40 is unchanged; this file is a separate artifact that points at it.

