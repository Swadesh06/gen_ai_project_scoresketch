# HumScribe Phase E item 4 — cross-platform Docker image.
#
# Default backend = HuggingFace `transformers.models.musicgen_melody`
# (no audiocraft, works on Windows-hosted Docker). The audiocraft backend
# remains importable for advanced users; B77 LoRA adapters trained against
# audiocraft's LM still load there.
#
# Build:    docker build -t humscribe:latest .
# Run:      docker run --gpus all -p 8501:8501 humscribe:latest
# CPU-only: docker run -e CUDA_VISIBLE_DEVICES= -p 8501:8501 humscribe:latest

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HUMSCRIBE_MUSICGEN_BACKEND=hf \
    HF_HOME=/root/.cache/huggingface \
    DEBIAN_FRONTEND=noninteractive

# System audio toolchain + Java for MV2H.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
        fluidsynth fluid-soundfont-gm sox ffmpeg \
        libsndfile1 libsox-fmt-all \
        default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. We use a pinned subset because the working env has
# 198 packages — most are transitive. The set below is what the pipeline
# imports directly.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy package source and install editable.
COPY humscribe /app/humscribe
COPY app /app/app
COPY scripts /app/scripts
COPY third_party /app/third_party
COPY pyproject.toml README.md /app/

# `.pth`-style install: humscribe importable from /app without pip install -e
ENV PYTHONPATH=/app

# Pre-cache the small / always-needed model weights so first-run is fast.
# (Large weights — YourMT3+, MusicGen-Melody-Large — are intentionally NOT
#  pre-baked here to keep image size manageable. They download on first call
#  to HF_HOME, which can be a mounted volume.)
RUN python -c "import torchcrepe, torch; \
    torchcrepe.predict(torch.zeros(1, 16000), 16000, 160, model='full', device='cpu')" \
    && python -c "from pesto import load_model; load_model('mir-1k_g7', step_size=10.0)" \
    || echo 'pre-cache phase had warnings (ok)'

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501"]
