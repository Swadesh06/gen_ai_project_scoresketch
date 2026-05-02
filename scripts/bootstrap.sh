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

# MAESTRO - piano reference, MIDI-only is fine for now (80 MB)
m = mirdata.initialize("maestro", data_home=f"{home}/maestro", version="3.0.0")
m.download(partial_download=["midi"])
print("maestro MIDI ready")

# MTG-QBH - casual humming reality check (150 MB)
q = mirdata.initialize("mtg_qbh", data_home=f"{home}/mtg_qbh")
q.download(); q.validate()
print("mtg_qbh ready")
PY

# ASAP - rhythm-quantization validation (50 MB scores+annotations only)
if [ ! -d "$HOME/datasets/asap" ]; then
    git clone https://github.com/CPJKU/asap-dataset.git "$HOME/datasets/asap"
fi
echo "asap ready"

# MIR-1K - PESTO sanity-check data (~500 MB)
if [ ! -d "$HOME/datasets/mir1k" ]; then
    cd "$HOME/datasets"
    wget -q --show-progress -O mir1k.zip "https://zenodo.org/records/3532216/files/MIR-1K.zip?download=1"
    unzip -q mir1k.zip -d mir1k && rm mir1k.zip
fi
echo "mir1k ready"
