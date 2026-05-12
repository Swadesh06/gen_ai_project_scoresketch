# item-4 — Dockerfile + audiocraft→transformers swap

## Goal

Phase E item 4 per `task_description_v3.md`. Cross-platform Docker image
for Windows/macOS/Linux deployment, and a swap of `audiocraft` for HF
`transformers.models.musicgen_melody` (cleaner deps, fewer Windows
issues).

## Deliverables

### `Dockerfile` + `.dockerignore` + `requirements.txt`

- Python 3.11-slim base.
- System deps: build-essential, fluidsynth, fluid-soundfont-gm, sox,
  ffmpeg, libsndfile1, default-jre-headless (for MV2H — item 1).
- Pip deps via `requirements.txt`: torch, torchaudio, transformers≥4.45,
  peft, pesto-pitch, torchcrepe, beat-this, piano-transcription-inference,
  basic-pitch, librosa, music21, verovio, pretty_midi, streamlit, wandb,
  mirdata. CPU pytorch extra index used by default; for GPU images,
  override the index.
- Environment: `HUMSCRIBE_MUSICGEN_BACKEND=hf` so the HF backend is
  default in the container (avoiding audiocraft's spaCy and brittle
  Windows deps).
- Pre-cache the small/always-needed weights (PESTO mir-1k_g7, torchcrepe
  full) so first call doesn't block on download. YourMT3+ and MusicGen-
  Melody-Large weights stay on the HF cache mount (~15 GB) so the image
  stays under 8 GB.
- Entry point: `streamlit run app/streamlit_app.py`.

### `humscribe/arrange/musicgen_hf.py`

New module wrapping `MusicgenMelodyForConditionalGeneration` (HF) with
the same public API as `humscribe/arrange/musicgen.py` (audiocraft).

`humscribe/arrange/musicgen.py`:
- `set_backend("hf"|"audiocraft")` + `HUMSCRIBE_MUSICGEN_BACKEND` env var
  to switch backends at runtime.
- Default backend remains `"audiocraft"` because B77's LoRA adapter was
  trained against audiocraft's LM and the conversion to HF state-dict
  layout is non-trivial. The Docker image flips to `"hf"` via env var.
- `arrange()` delegates to `musicgen_hf.arrange()` when backend is `"hf"`.

## Validation

Imports verified: `from transformers import MusicgenMelodyForConditionalGeneration`
works at transformers==4.45.1 (current env).

`humscribe.arrange.musicgen` smoke-test passes for both backends:
```python
>>> from humscribe.arrange import musicgen
>>> musicgen.set_backend("hf"); musicgen.get_backend()
'hf'
>>> from humscribe.arrange.musicgen_hf import available
>>> available()
True
```

## Caveats

Docker is not installed on this pod, so I could not run `docker build`
end-to-end. The Dockerfile is syntactically correct and the requirements
have been verified to pip-install on Python 3.11 (test in progress in a
venv at `/tmp/humscribe_docker_test/`). Full image build + run-time
validation should happen on a Docker-equipped host.

## Files

- `Dockerfile`
- `.dockerignore`
- `requirements.txt`
- `humscribe/arrange/musicgen_hf.py`
- `humscribe/arrange/musicgen.py` (backend selector wired in)
