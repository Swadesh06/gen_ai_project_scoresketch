# HumScribe native Windows install (PowerShell 7+).
#
# **Docker is the strongly recommended path for Windows users.** See
# docs/INSTALL.md section 1 for the Docker build/run. This script is a
# best-effort native install for users who cannot run Docker Desktop;
# native FluidSynth + Java + audio I/O on Windows have known sharp edges.
#
# Usage:
#   pwsh scripts/setup_windows.ps1                # default: venv at .venv\
#   pwsh scripts/setup_windows.ps1 -SkipChoco     # don't install via choco
#   pwsh scripts/setup_windows.ps1 -SkipSmoke     # skip the smoke test

param(
    [switch]$SkipChoco = $false,
    [switch]$SkipSmoke = $false
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")
$RepoRoot = (Get-Location).Path

function Log  ($msg) { Write-Host "[setup] $msg"  -ForegroundColor Cyan }
function Warn ($msg) { Write-Host "[warn ] $msg"  -ForegroundColor Yellow }
function Die  ($msg) { Write-Host "[fail ] $msg"  -ForegroundColor Red; exit 1 }

# ----------------------------------------------------------------------------
# 1. Preflight
# ----------------------------------------------------------------------------
Log "preflight"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    Warn "PowerShell $($PSVersionTable.PSVersion) — pwsh 7+ recommended"
}

# Big visible warning that Docker is the recommended path.
Write-Host ""
Write-Host "  >> Docker is the recommended HumScribe install path on Windows. <<"
Write-Host "     If you have Docker Desktop, prefer:"
Write-Host "         docker build -t humscribe ."
Write-Host "         docker run -p 8501:8501 humscribe"
Write-Host "     This native script is a best-effort fallback."
Write-Host ""

# ----------------------------------------------------------------------------
# 2. Chocolatey + system audio toolchain
# ----------------------------------------------------------------------------
if ($SkipChoco) {
    Warn "skipping chocolatey step (-SkipChoco)"
} else {
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Warn "Chocolatey not detected; installing it (admin shell required)"
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol =
            [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString(
            "https://community.chocolatey.org/install.ps1"))
    }
    $chocoPkgs = @("python311", "fluidsynth", "ffmpeg", "sox.portable", "temurin17")
    Log "choco install $($chocoPkgs -join ' ')"
    choco install -y @chocoPkgs
}

# ----------------------------------------------------------------------------
# 3. Python: prefer python3.11
# ----------------------------------------------------------------------------
$Py = $null
foreach ($candidate in @("python3.11", "python", "py")) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            $ver = & $cmd.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
            if ($ver.Trim() -eq "3.11") {
                $Py = $cmd.Source
                break
            }
        } catch { }
    }
}
if (-not $Py) {
    $Py = (Get-Command python -ErrorAction SilentlyContinue).Source
    Warn "Python 3.11 not found; falling back to $Py"
}
if (-not $Py) { Die "no Python interpreter found" }
Log "using $Py"

# ----------------------------------------------------------------------------
# 4. venv
# ----------------------------------------------------------------------------
$VenvDir = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Log "creating venv at .venv\"
    & $Py -m venv $VenvDir
}
$ActivatePs1 = Join-Path $VenvDir "Scripts\Activate.ps1"
. $ActivatePs1
Log "active python: $(Get-Command python | Select-Object -ExpandProperty Source)"

# ----------------------------------------------------------------------------
# 5. Pip install
# ----------------------------------------------------------------------------
Log "upgrading pip"
python -m pip install --upgrade pip *> $null

Log "installing requirements.txt"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu `
    -r requirements.txt
python -m pip install cairosvg *> $null

# ----------------------------------------------------------------------------
# 6. Pre-cache small model weights
# ----------------------------------------------------------------------------
Log "pre-caching small model weights"
$preCache = @"
import torch, torchcrepe
torchcrepe.predict(torch.zeros(1, 16000), 16000, 160, model='tiny', device='cpu')
print('[pre-cache] torchcrepe tiny  ok')
try:
    from pesto import load_model
    load_model('mir-1k_g7', step_size=10.0)
    print('[pre-cache] pesto mir-1k_g7 ok')
except Exception as exc:
    print(f'[pre-cache] pesto skipped: {exc}')
"@
try {
    python -c "$preCache"
} catch {
    Warn "pre-cache had warnings (non-fatal)"
}

# ----------------------------------------------------------------------------
# 7. Smoke test
# ----------------------------------------------------------------------------
if ($SkipSmoke) {
    Warn "skipping smoke test (-SkipSmoke)"
} else {
    Log "smoke test 1/2 - package import"
    python -c "from humscribe.pipeline import transcribe; print('[smoke] import ok')"

    if (Test-Path "app/demos/demo_1_vocadito_S1.wav") {
        Log "smoke test 2/2 - pipeline transcription"
        New-Item -ItemType Directory -Force -Path "outputs/smoke" | Out-Null
        python scripts/smoke_transcribe.py
    } else {
        Warn "demo clip not present; transcription smoke skipped"
    }
}

Write-Host ""
Write-Host "[ done ]  HumScribe environment ready (Windows, best-effort)." -ForegroundColor Green
Write-Host ""
Write-Host "Known sharp edges on native Windows:"
Write-Host "  - FluidSynth: GM SoundFont path must be set manually for synth"
Write-Host "  - MV2H Java tool: needs JAVA_HOME pointing at the temurin install"
Write-Host "  - librsvg: not installed; SVG to PNG conversion may need Inkscape"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  - streamlit run app/streamlit_app.py"
Write-Host "  - python scripts/generate_pitch_figures.py"
Write-Host ""
Write-Host "If anything fails on native Windows, fall back to Docker (recommended)."
Write-Host ""
