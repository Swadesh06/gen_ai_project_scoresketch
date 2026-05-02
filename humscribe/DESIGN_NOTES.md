# HumScribe v3.2 — Design Notes

The shipped spec (`scoresketch.md`) is the v3.2 addendum. It assumes v3 + v3.1
architecture is specified elsewhere and only describes deltas: three new
datasets, three new eval scripts, and a bootstrap diff. Almost everything below
is inferred from the call sites in §D, the access table in §A, the validation
gates, and the stage references ("Stage 4", "Stage 5", "Stage 2-B.1") that appear
without their definitions in scope. Every inference is listed here so the human
can correct anything that drifted from intent.

## Pipeline stage map (inferred)

The eval scripts and prose imply a multi-stage pipeline. The numbering reused
below is consistent with the spec's "Stage 4 = beat tracking, Stage 5 = rhythm
quantization" references.

- **Stage 1 — Audio I/O**: load, mono-mix, resample to a model-appropriate rate.
- **Stage 2-A — Pitch tracking (humming branch)**: PESTO (default) or CREPE.
  Returns frame-level f0 + voicing.
- **Stage 2-B — Instrument transcription (instrument branch)**: ByteDance piano
  for piano input, Basic Pitch otherwise. Returns NoteEvents directly with
  onset/offset/pitch — skips Stage 3.
- **Stage 3 — Note segmentation (humming branch only)**: HMM/voicing-driven
  segmenter that converts a continuous f0 contour into discrete NoteEvents.
- **Stage 4 — Beat tracking**: beat_this returns `(beats, downbeats, bpm)`.
- **Stage 5 — Rhythm quantization**: Cemgil–Kappen DP that snaps onsets and
  offsets to a 12-tatum-per-beat grid (12 chosen because the eval script does
  `(q_off - q_on) / 12.0` to get quarters). Outputs integer tatum indices.
- **Stage 6 — Score render**: build a music21 Stream from quantized notes +
  beats + meter, write MusicXML, then SVG. SVG rendering is best-effort: prefer
  music21's MusicXML→SVG via LilyPond/MuseScore if installed, otherwise a
  minimal hand-rolled SVG fallback so `r.svg` is never empty (see `score.py`).

## `PipelineConfig` schema (inferred)

The §D scripts call `PipelineConfig(input_kind="humming", mode=mode)`. So the
two required fields are `input_kind` and `mode`. Inferred extras:

- `input_kind: Literal["humming", "instrument", "piano", "guitar"]` — picks
  branch A (pitch+segmenter) vs branch B (transcriber); piano/guitar are
  instrument-branch sub-types that select the model.
- `mode: Literal["soft", "medium", "hard"]` — tightness of onset/offset
  detection and rhythm quantization. Switch handling is exhaustive in
  `ModeConfig.for_mode()`.
- `pitch_model: Literal["pesto", "crepe"]` — default `"pesto"`.
- `transcriber: Literal["bytedance_piano", "basic_pitch"]` — default chosen
  from `input_kind`.
- `tatums_per_beat: int = 12` — fixed by the eval script's tatum→quarter math.
- `sample_rate: int = 22050` — model-friendly default; load_audio resamples.
- `render_svg: bool = True`.

## Mode semantics (inferred from "soft | medium | hard")

| Field | soft | medium | hard |
|---|---|---|---|
| voicing_threshold (PESTO confidence) | 0.30 | 0.50 | 0.70 |
| min_note_seconds | 0.06 | 0.10 | 0.15 |
| onset_merge_seconds | 0.08 | 0.05 | 0.03 |
| dp_offgrid_penalty | 0.5 | 1.0 | 2.0 |
| pitch_smooth_window | 7 | 5 | 3 |

Soft is the humming-friendly default; hard is for clean studio audio. These
numbers are educated starting points — they can be tuned against Vocadito and
MTG-QBH on the GPU pod without changing the public API.

## `NoteEvent` schema

Defined in `humscribe/notes.py`:

```python
@dataclass
class NoteEvent:
    onset_s: float
    offset_s: float
    pitch_hz: float | None = None
    pitch_midi: int | None = None
    velocity: int = 80
    confidence: float = 1.0
```

`onset_s` and `offset_s` are spec-mandated by the §D.1 eval script.
`pitch_midi` is needed by the score builder; `pitch_hz` is what pitch trackers
emit natively. Either field may be set.

## Module call-site contracts (locked by §D scripts)

These signatures are non-negotiable; the eval scripts depend on them verbatim:

- `humscribe.beat.beat_this_track.track_beats_beat_this(audio_path: str)`
  → `(beats: np.ndarray, downbeats: np.ndarray, bpm: float)`
- `humscribe.rhythm.viterbi_quantize.viterbi_quantize_rhythm(onsets, offsets, beats)`
  → `(q_on: np.ndarray[int], q_off: np.ndarray[int])` in tatum units
  (12 tatums per beat).
- `humscribe.instrument.piano.transcribe_piano(audio_path: str)`
  → `list[NoteEvent]` (with `.onset_s`, `.offset_s`).
- `humscribe.pitch.pesto_track.track_pitch_pesto(audio: np.ndarray, sr: int)`
  → `(times: np.ndarray, hz: np.ndarray, voicing: np.ndarray)`.
- `humscribe.pipeline.transcribe(audio_path, cfg) -> TranscribeResult` with
  `.svg: str`, `.n_notes: int`, `.bpm: float`.
- `humscribe.config.PipelineConfig(input_kind, mode, ...)`.

## Cemgil–Kappen DP — chosen formulation

Classical Cemgil–Kappen rhythm quantization minimises a cost combining (a) the
Gaussian onset-deviation log-likelihood at tatum positions and (b) a transition
cost on duration changes. We implement it as a 1-D DP over onsets only; offsets
are quantized independently to the next legal tatum at or after the rounded
offset. Tatum grid: 12 per beat (covers eighth, eighth-triplet, sixteenth, and
their dotted variants).

Cost per onset at tatum index t:
    `-log N(t/12 - onset_in_beats; 0, sigma)` (Gaussian fit to mode's onset jitter)
Plus transition penalty on consecutive intervals to discourage off-grid drift:
    `dp_offgrid_penalty * (1 - cos(pi * frac_part))`.

Standard forward DP, integer state space, traceback at the end.

## Score rendering

`humscribe/score.py` builds a music21 `Stream` from quantized NoteEvents and a
detected time signature (defaulting to 4/4 when downbeats are sparse). MusicXML
output is direct. SVG output tries music21's `Stream.write('musicxml.svg')` if
LilyPond/MuseScore is reachable; on failure it returns a minimal pure-SVG
piano-roll fallback so the eval script's `r.svg.write_text(...)` never crashes.

## What's NOT here that the spec might assume from v3/v3.1

- Detailed thresholds for the original Stage 1–6 gates (only §B.1 thresholds for
  the new ASAP gate are stated explicitly).
- The exact HMM topology for the humming-path note segmenter — ours is a
  pragmatic median-filter + voicing-gated state-change segmenter, which is
  cheaper than Viterbi and gives reasonable results on Vocadito-style input.
  Replace with a proper HMM if the v3/v3.1 doc demands it.
- The "Soft / Medium / Hard" mode constants. Numbers above are first-pass
  defaults.
- Choice of SoundFont path for FluidSynth-rendered MIDI inputs (used only when
  ASAP gives MIDI without rendered WAV — `audio_io.maybe_render_midi()` falls
  back to `FluidR3_GM.sf2` at the standard Debian apt path).

If anything in the v3/v3.1 source contradicts the above, prefer that spec and
update this file.

## Known gotchas hit during the CPU build

These are real findings from setup that affect the GPU phase:

- **`mtg_qbh` is not in `mirdata` 1.0.0** (latest as of this build). The spec
  §B.2 claims `mirdata.initialize("mtg_qbh", ...)` works; it does not. The
  bootstrap script (kept verbatim per §C) and `scripts/eval_mtg_qbh_visual.py`
  (kept verbatim per §D.3) will both fail at the MTG-QBH step. Fix options:
  download MTG-QBH directly from Zenodo record `1290712` and write a thin
  custom loader, or pin an older mirdata that exposes it (none of the listed
  versions on PyPI register `mtg_qbh` either — verify by `mirdata.list_datasets()`).
- **Bootstrap not run in the CPU phase.** `bash scripts/bootstrap.sh` is a
  ~700 MB download with no GPU dependency, but it would fail mid-run on the
  `mtg_qbh` line above and partially populate `~/datasets`. Re-run after the
  loader fix on the GPU pod.
- **TensorFlow 2.15 emits cuDNN-already-registered warnings on import.** Cosmetic
  and harmless — the import succeeds and the package is usable. Comes from
  basic-pitch pulling in the GPU-enabled TF wheel (only one available on PyPI).
- **`piano_transcription_inference` 0.0.6 first call downloads ~330 MB from
  Zenodo to `~/.cache/torch/checkpoints/`.** Class instantiation triggers it
  (not import). On CPU this is fine but slow; on GPU it's the right thing.
- **`beat_this` `File2Beats` first call downloads ~70 MB.** Same shape — class
  instantiation triggers it.
- **PESTO / Basic Pitch / torchcrepe** all bundle weights (PESTO, BP) or
  download to `~/.cache` lazily (torchcrepe). No license-request friction.
- **Torch is the CPU wheel.** Reinstall with the correct CUDA wheel after the
  pod swap, e.g. `pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio` —
  but check `nvidia-smi` driver version first to pick the matching index.

## Post-build fix: `humscribe` package install (resolved)

The original CPU build produced source files but never installed the `humscribe`
package into the env. `import humscribe` only worked when `cwd` happened to
contain the `humscribe/` folder; from any other directory it raised
`ModuleNotFoundError`. All three Python eval scripts in `scripts/` would have
failed on their first import line on the GPU phase.

Fix applied (post-handoff, before GPU swap):

1. Added `pyproject.toml` at `/workspace/swadesh/gen_ai_project_scoresketch/`
   (setuptools backend, package name `humscribe`, finds the `humscribe*`
   subtree).
2. Tried `pip install -e .` first — it works but `conda-pack` refuses to pack
   envs with editable installs (`CondaPackError: Cannot pack an environment
   with editable packages installed`), because the editable install stores a
   path on local disk that won't exist on a fresh pod.
3. Replaced the editable install with a plain `.pth` file at
   `$CONDA_PREFIX/lib/python3.11/site-packages/humscribe.pth` containing the
   single line `/workspace/swadesh/gen_ai_project_scoresketch`. Python's
   `site` module reads `*.pth` files at interpreter startup and treats every
   line as an extra `sys.path` entry. `conda-pack` is happy with `.pth` files
   (they're plain text, not editable installs), and the path is on the
   persistent `/workspace` volume so it stays valid across pod restarts.
4. Repacked: `bash /workspace/scripts/pack_env.sh humscribe`. Tarball at
   `/workspace/env-archives/humscribe.tar.gz` now includes the `.pth` file.

Verification: `cd / && python -c "import humscribe; from humscribe.pipeline
import transcribe"` succeeds. Same for all five `humscribe.*` imports the eval
scripts depend on.

Properties of this fix that matter going forward:
- Edits to `humscribe/*.py` are picked up live, no reinstall needed (same
  property as `pip install -e`).
- On a fresh pod, after `bash /workspace/scripts/startup.sh humscribe`, imports
  work everywhere out of the box.
- If the project is ever moved off `/workspace/swadesh/gen_ai_project_scoresketch`,
  update the single line in `humscribe.pth` and repack.

## What's left to run on the GPU phase

The package-install issue (see "Post-build fix" above) is already resolved — do
not redo it. Remaining steps:

1. Reinstall torch with the matching CUDA wheel; rerun `pip check`.
2. Patch the `mtg_qbh` loader (or replace the script's `mirdata.initialize` call).
3. `bash scripts/bootstrap.sh` to pull all 5 datasets.
4. `python scripts/eval_mir1k_pitch_sanity.py` (Stage 2-B.1 PESTO sanity gate).
5. `python scripts/eval_asap_rhythm.py` (Stage 4 + 5 gates on Bach BWV 846).
6. `python scripts/eval_mtg_qbh_visual.py --modes soft,medium` (Phase-2 reality
   check; eyeball the SVGs).
7. Repack env after any package additions or torch swap:
   `bash /workspace/scripts/pack_env.sh humscribe`.
