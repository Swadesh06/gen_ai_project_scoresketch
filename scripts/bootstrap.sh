#!/usr/bin/env bash
# HumScribe v3.2 bootstrap. Downloads Vocadito + MAESTRO + ASAP + MTG-QBH + MIR-1K
# into ~/datasets. Idempotent.
set -e

mkdir -p "$HOME/datasets"

# 6. download datasets (Vocadito + MAESTRO + ASAP + MTG-QBH + MIR-1K)
echo "=== Downloading evaluation datasets ==="
python - <<'PY'
import mirdata, os, subprocess
home = os.path.expanduser("~/datasets")

# Vocadito - primary humming eval (60 MB)
v = mirdata.initialize("vocadito", data_home=f"{home}/vocadito")
v.download(); v.validate()
print("vocadito ready")

# MAESTRO - piano reference, MIDI-only (mirdata 1.0.0 ships v2.0.0; the spec
# named v3.0.0 but the older index is sufficient for our reference use)
m = mirdata.initialize("maestro", data_home=f"{home}/maestro", version="2.0.0")
m.download(partial_download=["midi"])
print("maestro MIDI ready")

PY

# MTG-QBH - casual humming reality check (150 MB)
# mirdata 1.0.0 doesn't register mtg_qbh; we use a Zenodo-direct loader.
python - <<'PY'
import os
from humscribe.datasets.mtg_qbh import MTGQBH
home = os.path.expanduser("~/datasets")
q = MTGQBH(data_home=f"{home}/mtg_qbh")
q.download(); q.validate()
print("mtg_qbh ready")
PY

# ASAP - rhythm-quantization validation (50 MB scores+annotations only)
if [ ! -d "$HOME/datasets/asap" ]; then
    git clone https://github.com/CPJKU/asap-dataset.git "$HOME/datasets/asap"
fi
echo "asap ready"

# MIR-1K - PESTO sanity-check data (~1 GB).
# The Zenodo record (3532216) is metadata-only and points at mirlab.org, which
# returns 404. We use the HF mirror (AnhP/Mir-1k-use-DJCM-training).
if [ ! -d "$HOME/datasets/mir1k/MIR-1K" ]; then
    python - <<'PY'
import os, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download
out = Path("~/datasets/mir1k").expanduser()
out.mkdir(parents=True, exist_ok=True)
zp = hf_hub_download(
    repo_id="AnhP/Mir-1k-use-DJCM-training",
    filename="MIR-1K.zip",
    repo_type="dataset",
    cache_dir=os.environ.get("HF_HOME", None),
)
with zipfile.ZipFile(zp) as z:
    z.extractall(out)
print("mir1k extracted to", out)
PY
fi
echo "mir1k ready"
