#!/usr/bin/env bash
# HumScribe native macOS install (Homebrew + Python 3.11 venv).
#
# **Best-effort, untested on the development host** (Linux). The Docker
# path is the recommended cross-platform install for macOS — see
# docs/INSTALL.md section 1.
#
# Usage:
#   bash scripts/setup_macos.sh                # python venv at .venv/
#   bash scripts/setup_macos.sh --skip-brew    # skip brew install step
#   bash scripts/setup_macos.sh --skip-smoke   # skip the smoke test

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

SKIP_BREW=0
SKIP_SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --skip-brew)  SKIP_BREW=1 ;;
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
# 1. Preflight
# ----------------------------------------------------------------------------
log "preflight"
[[ "$(uname -s)" == "Darwin" ]] || die "this script is macOS-only (uname=$(uname -s))"

# ----------------------------------------------------------------------------
# 2. Homebrew + system audio toolchain
# ----------------------------------------------------------------------------
if [[ "$SKIP_BREW" -eq 1 ]]; then
    warn "skipping brew step (--skip-brew)"
else
    if ! command -v brew >/dev/null 2>&1; then
        warn "Homebrew not detected — installing it"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    log "brew install: python@3.11 fluidsynth fluid-r3-soundfont sox ffmpeg openjdk@17 librsvg"
    brew install python@3.11 fluidsynth fluid-r3-soundfont sox ffmpeg openjdk@17 librsvg

    # OpenJDK needs to be on PATH for MV2H to run. Tell the user explicitly
    # rather than touching ~/.zshrc silently.
    if ! /usr/bin/which java >/dev/null 2>&1; then
        warn "java is not on PATH — add this to your shell rc and re-source:"
        warn '    export PATH="$(brew --prefix)/opt/openjdk@17/bin:$PATH"'
    fi
fi

# ----------------------------------------------------------------------------
# 3. Python: prefer python3.11 from brew, fall back to system python3
# ----------------------------------------------------------------------------
PY=""
for candidate in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
[[ -n "$PY" ]] || die "no Python interpreter found; install python@3.11"
log "using $PY ($($PY --version 2>&1))"

# ----------------------------------------------------------------------------
# 4. venv
# ----------------------------------------------------------------------------
if [[ ! -d ".venv" ]]; then
    log "creating venv at .venv/"
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
log "active python: $(which python)"

# ----------------------------------------------------------------------------
# 5. Pip install
# ----------------------------------------------------------------------------
log "upgrading pip"
python -m pip install --upgrade pip >/dev/null

log "installing requirements.txt"
# macOS arm64 has the MPS-friendly PyTorch wheel as the default on PyPI; we
# don't need the CPU index URL.
python -m pip install -r requirements.txt
python -m pip install cairosvg >/dev/null

# ----------------------------------------------------------------------------
# 6. Pre-cache small model weights
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
# 7. Smoke test
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

cat <<EOF

\033[1;32m[ done ]\033[0m  HumScribe environment ready (macOS, best-effort).

Next steps:
  - launch UI:    streamlit run app/streamlit_app.py
  - regen figs:   python scripts/generate_presentation_figures.py

Docker path is the **recommended** macOS install — see docs/INSTALL.md.

EOF
