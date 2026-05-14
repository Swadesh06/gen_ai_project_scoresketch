#!/usr/bin/env bash
# HumScribe native Linux install (Ubuntu/Debian).
# Usage:
#   bash scripts/setup_linux.sh                # uses a python venv at .venv/
#   bash scripts/setup_linux.sh --conda        # uses conda env "humscribe"
#   bash scripts/setup_linux.sh --skip-apt     # skip system pkg install
#   bash scripts/setup_linux.sh --skip-smoke   # skip the final smoke test
#
# Re-runnable. Each phase is idempotent — re-running after a partial
# install picks up where it left off.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

USE_CONDA=0
SKIP_APT=0
SKIP_SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --conda)      USE_CONDA=1 ;;
        --skip-apt)   SKIP_APT=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $arg" >&2 ; exit 2 ;;
    esac
done

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[fail ]\033[0m %s\n" "$*" >&2 ; exit 1; }

# ----------------------------------------------------------------------------
# 1. Preflight: confirm we're on Linux + apt is available
# ----------------------------------------------------------------------------
log "preflight"
[[ "$(uname -s)" == "Linux" ]] || die "this script is Linux-only (uname=$(uname -s))"
command -v apt-get >/dev/null 2>&1 || warn "apt-get not found — system deps must be installed manually"

# ----------------------------------------------------------------------------
# 2. System audio toolchain (fluidsynth + soundfont + sox + ffmpeg + Java)
# ----------------------------------------------------------------------------
if [[ "$SKIP_APT" -eq 1 ]]; then
    warn "skipping apt step (--skip-apt)"
else
    APT_PKGS=(
        build-essential ca-certificates curl git
        fluidsynth fluid-soundfont-gm
        sox ffmpeg
        libsndfile1 libsox-fmt-all
        default-jre-headless
        librsvg2-bin
    )
    SUDO=""
    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            SUDO="sudo"
        else
            warn "no passwordless sudo; please run:  sudo apt-get install -y ${APT_PKGS[*]}"
            warn "skipping apt step automatically"
            SKIP_APT=1
        fi
    fi
    if [[ "$SKIP_APT" -ne 1 ]]; then
        log "installing system packages via apt-get"
        $SUDO apt-get update -y
        $SUDO apt-get install -y --no-install-recommends "${APT_PKGS[@]}"
    fi
fi

# ----------------------------------------------------------------------------
# 3. Python env: prefer Python 3.11 (project pin)
# ----------------------------------------------------------------------------
PY=""
for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done
if [[ -z "$PY" ]]; then
    warn "Python 3.11 not found on PATH"
    if command -v python3 >/dev/null 2>&1; then
        PY="python3"
        warn "falling back to $($PY --version 2>&1)"
    else
        die "no Python interpreter found"
    fi
fi
log "using $PY ($($PY --version 2>&1))"

# ----------------------------------------------------------------------------
# 4. Create / activate venv or conda env
# ----------------------------------------------------------------------------
if [[ "$USE_CONDA" -eq 1 ]]; then
    command -v conda >/dev/null 2>&1 || die "--conda requested but conda not on PATH"
    if ! conda env list | awk '{print $1}' | grep -qx "humscribe"; then
        log "creating conda env 'humscribe' with python=3.11"
        conda create -y -n humscribe python=3.11
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate humscribe
else
    if [[ ! -d ".venv" ]]; then
        log "creating venv at .venv/"
        "$PY" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
log "active python: $(which python)"

# ----------------------------------------------------------------------------
# 5. Pip install runtime deps
# ----------------------------------------------------------------------------
log "upgrading pip"
python -m pip install --upgrade pip >/dev/null

log "installing requirements.txt"
# Use the CPU PyTorch wheel index so people without CUDA get the right wheel.
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# cairosvg is used by some figure scripts; not in requirements.txt yet but
# harmless to install for parity with the Streamlit + figure paths.
python -m pip install cairosvg >/dev/null

# ----------------------------------------------------------------------------
# 6. Pre-cache small model weights (PESTO + CREPE tiny call)
# ----------------------------------------------------------------------------
log "pre-caching small model weights"
python - <<'PY' || warn "pre-cache had warnings (non-fatal)"
import torch, torchcrepe
torchcrepe.predict(torch.zeros(1, 16000), 16000, 160, model="tiny", device="cpu")
print("[pre-cache] torchcrepe tiny  ok")
try:
    from pesto import load_model
    load_model("mir-1k_g7", step_size=10.0)
    print("[pre-cache] pesto mir-1k_g7 ok")
except Exception as exc:
    print(f"[pre-cache] pesto skipped: {exc}")
PY

# ----------------------------------------------------------------------------
# 7. Smoke test: import + transcribe a 13 s humming demo
# ----------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" -eq 1 ]]; then
    warn "skipping smoke test (--skip-smoke)"
else
    log "smoke test 1/2  — package import"
    python -c "from humscribe.pipeline import transcribe; print('[smoke] import ok')"

    if [[ -f "app/demos/demo_1_vocadito_S1.wav" ]]; then
        log "smoke test 2/2  — pipeline transcription"
        mkdir -p outputs/smoke
        python scripts/smoke_transcribe.py
    else
        warn "demo clip not present; transcription smoke skipped"
    fi
fi

# ----------------------------------------------------------------------------
# 8. Success banner
# ----------------------------------------------------------------------------
cat <<EOF

\033[1;32m[ done ]\033[0m  HumScribe environment ready.

Next steps:
  - launch UI:           streamlit run app/streamlit_app.py
  - rebuild figures:     python scripts/generate_presentation_figures.py
  - rebuild MAESTRO svg: python scripts/revert_maestro_demo.py
  - regen 4 slide figs:  python scripts/generate_pitch_figures.py

Docker path (recommended for cross-platform):
  docker build -t humscribe . && docker run -p 8501:8501 humscribe

EOF
