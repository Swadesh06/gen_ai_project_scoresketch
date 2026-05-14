# HumScribe install guide

HumScribe runs on Linux, macOS, and Windows. The **Docker** path is the
recommended cross-platform install — the same image is what the
Dockerfile-driven CI image uses, and it sidesteps the FluidSynth +
soundfont + Java toolchain headache that native Windows in particular
suffers from.

## 1. Quick start — Docker (recommended for all platforms)

Prerequisites: Docker Desktop (Windows / macOS) or Docker Engine
(Linux). For GPU inference, also install the NVIDIA container toolkit
on Linux.

```bash
# from the repo root
docker build -t humscribe .

# CPU-only:
docker run -p 8501:8501 humscribe

# with GPU (Linux + NVIDIA container toolkit):
docker run --gpus all -p 8501:8501 humscribe
```

Open <http://localhost:8501> in a browser. You should see the
HumScribe Streamlit UI with a Transcribe tab and an Arrange tab. The
five pre-recorded humming demos load directly from the dropdown — no
upload required.

The Dockerfile uses the HuggingFace `transformers.models.musicgen`
backend by default (`HUMSCRIBE_MUSICGEN_BACKEND=hf`); the heavyweight
audiocraft dependency is **not** installed in the image, which keeps
the build size manageable and is the only path that works on
Windows-hosted Docker.

## 2. Linux native install

Tested on Ubuntu 22.04 LTS (jammy). Other apt-based distros should
work; RPM-based distros need to translate the package names.

```bash
bash scripts/setup_linux.sh
```

The script does:

1. `apt-get install` for the audio toolchain (fluidsynth + GM
   soundfont, sox, ffmpeg, libsndfile, default-jre for MV2H,
   librsvg-bin for SVG→PNG).
2. Creates a Python 3.11 venv at `.venv/` (use `--conda` to use a
   conda env named `humscribe` instead).
3. `pip install -r requirements.txt`.
4. Pre-caches the small PESTO and torchcrepe-tiny weights.
5. Runs the package import + a pipeline smoke transcription on
   `app/demos/demo_1_vocadito_S1.wav`.

System requirements:

- Python 3.11 (the project pin); Python 3.10/3.12 may work but is
  untested.
- Java 11+ JRE for MV2H eval. The setup script installs
  `default-jre-headless`.
- ~10 GB free disk for the venv + pre-cached small models.
- GPU is optional. CPU inference works for transcription but Stage 7
  (MusicGen) is impractically slow without a GPU.

Useful flags:

- `--conda` use a conda env instead of a venv.
- `--skip-apt` skip the apt step (use if you already have the tools).
- `--skip-smoke` skip the final transcription smoke test.

## 3. macOS native install

```bash
bash scripts/setup_macos.sh
```

**Best-effort, untested on the development host (Linux).** The Docker
install is preferred. The script does:

1. `brew install python@3.11 fluidsynth fluid-r3-soundfont sox ffmpeg
   openjdk@17 librsvg` (installs Homebrew first if missing).
2. Creates `.venv/` and pip-installs `requirements.txt`.
3. Pre-cache + smoke test identical to the Linux flow.

You must put OpenJDK on PATH for MV2H to work — the script prints the
exact `export PATH=...` line to add to `~/.zshrc`.

Apple-silicon Macs use the default PyPI PyTorch wheel (MPS-friendly);
Intel Macs fall back to CPU.

## 4. Windows install

**Docker is strongly recommended for Windows users.** The native
PowerShell path is provided as a fallback for users who cannot run
Docker Desktop.

Docker (recommended):

```powershell
docker build -t humscribe .
docker run -p 8501:8501 humscribe
```

Native PowerShell:

```powershell
pwsh scripts/setup_windows.ps1
```

The script uses Chocolatey to install Python 3.11, FluidSynth, FFmpeg,
SoX, and the Temurin OpenJDK 17, then creates `.venv\` and pip-installs
requirements.

Known sharp edges on native Windows:

- **FluidSynth** on Windows requires its DLL to be discoverable on
  PATH. Chocolatey usually places it correctly, but the GM SoundFont
  is not bundled — set the `SOUND_FONT` env var if synth fails.
- **MV2H Java**: the `default-jre` package isn't on Windows; the
  script installs Temurin 17 via Chocolatey. After install, ensure
  `java -version` works in a fresh shell (PATH change requires
  reopen).
- **librsvg / cairosvg**: SVG→PNG conversion needs librsvg. The
  Chocolatey package set does not include it; Docker does. If
  `scripts/generate_pitch_figures.py` fails on the SVG conversion,
  install Inkscape and convert manually.

Useful switches:

- `-SkipChoco` skip the Chocolatey install step.
- `-SkipSmoke` skip the final transcription smoke test.

## 5. Verify installation

After any install path completes, run:

```bash
# package import
python -c "from humscribe.pipeline import transcribe; print('ok')"

# end-to-end smoke transcription (writes to outputs/smoke/)
python scripts/smoke_transcribe.py
```

Expected output (~30 s on GPU, ~3 min on CPU):

```
notes=61  bpm=115.38
svg_exists=True  mxl_exists=True
svg=outputs/smoke/smoke.svg
mxl=outputs/smoke/smoke.musicxml
```

If both `notes=` and `svg_exists=True` print, the install is good.

## 6. Common errors and fixes

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | requirements.txt not pip-installed into active env | `pip install streamlit cairosvg` or re-run `setup_linux.sh` |
| `RuntimeError: fluidsynth not found` | FluidSynth not on PATH | Linux: `sudo apt-get install fluidsynth fluid-soundfont-gm`. macOS: `brew install fluidsynth fluid-r3-soundfont`. Windows: Docker is easier |
| `Java not found` when MV2H eval runs | JRE not installed or not on PATH | install OpenJDK 11+ (Temurin on Windows, `default-jre-headless` on Linux, `openjdk@17` on macOS) |
| `OSError: [Errno 98] Address already in use` on Streamlit | port 8501 occupied | `streamlit run app/streamlit_app.py --server.port 8502` or kill the prior process |
| CUDA mismatch (`undefined symbol __cudaPushCallConfiguration`) | system CUDA != PyTorch CUDA | reinstall PyTorch matching system CUDA, or pass `--extra-index-url https://download.pytorch.org/whl/cpu` for CPU-only |
| Verovio `SMuFL glyph U+E260 not found` warnings during render | benign Verovio warning about Bravura fallback | ignore — does not affect SVG output |
| SVG→PNG step fails in figure scripts | `cairosvg` not installed (or librsvg missing on Windows) | `pip install cairosvg`, or use `rsvg-convert` (Linux/macOS) via subprocess |

## 7. Running the demo

After install, two demo paths exist:

**Streamlit UI** (recommended for live presentation):

```bash
streamlit run app/streamlit_app.py
# then open http://localhost:8501
```

Use the "Transcribe" tab and pick one of the five pre-recorded demos
in the dropdown — no upload required. The result renders inline as an
SVG with a download button for MusicXML. Switch to the "Arrange" tab
to run Stage 7 MusicGen arrangement on the just-transcribed clip.

**Script-only path**:

```bash
# 13 s humming demo, writes to outputs/smoke/
python scripts/smoke_transcribe.py

# regenerate the four slide figures + the seven journey figures
python scripts/generate_pitch_figures.py
python scripts/generate_presentation_figures.py

# revert the MAESTRO chamber demo to the clean render_tpb=8 state
python scripts/revert_maestro_demo.py
```

For the live presentation, the **Docker → Streamlit** path is the most
reliable: it builds once, runs anywhere, and avoids the FluidSynth and
MV2H Java pitfalls.
