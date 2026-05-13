#!/usr/bin/env bash
# Phase G G-17 — docker actual-build verification harness.
#
# The agent ships the harness; the user runs it on a host with docker
# installed (the sandbox has no docker binary). Strict pass when this
# script exits 0 on the user's machine.
#
# Usage:  bash scripts/g17_docker_verify.sh [--no-run]
#
# Checks (in order):
#   1. docker present
#   2. docker daemon reachable
#   3. Dockerfile + .dockerignore + requirements.txt + humscribe/ exist
#   4. `docker build -t humscribe:phase-g .` succeeds
#   5. Smoke: `docker run --rm humscribe:phase-g python -c "import humscribe; print('ok')"`
#   6. (skipped if --no-run) `docker run -p 8501:8501 ... &` and curl localhost:8501
set -euo pipefail

cd "$(dirname "$0")/.."

NO_RUN=0
for arg in "$@"; do
  if [ "$arg" = "--no-run" ]; then NO_RUN=1; fi
done

echo "== g17 docker verify =="
echo "cwd: $(pwd)"

# 1. docker present?
if ! command -v docker >/dev/null 2>&1; then
  echo "FAIL: docker binary not on PATH. Install Docker Desktop or docker-ce, then retry."
  exit 1
fi
echo "ok: docker found at $(command -v docker)"

# 2. docker daemon reachable?
if ! docker info >/dev/null 2>&1; then
  echo "FAIL: docker daemon not running. Start Docker Desktop or 'sudo systemctl start docker'."
  exit 1
fi
echo "ok: docker daemon reachable"

# 3. required files present?
for f in Dockerfile .dockerignore requirements.txt; do
  if [ ! -f "$f" ]; then
    echo "FAIL: missing required file: $f"
    exit 1
  fi
done
for d in humscribe app scripts third_party; do
  if [ ! -d "$d" ]; then
    echo "FAIL: missing required directory: $d"
    exit 1
  fi
done
echo "ok: required files + directories present"

# 4. docker build
echo "== docker build =="
docker build -t humscribe:phase-g .
echo "ok: docker build succeeded"

# 5. import smoke
echo "== import smoke =="
docker run --rm humscribe:phase-g python -c "import humscribe; from humscribe.eval.mv2h import compute_mv2h; print('humscribe imports ok')"
echo "ok: humscribe imports inside the container"

# 6. optional: streamlit smoke
if [ "$NO_RUN" -eq 0 ]; then
  echo "== streamlit smoke =="
  CONTAINER_ID=$(docker run -d -p 8501:8501 humscribe:phase-g)
  sleep 8
  if curl --silent --output /dev/null --fail http://localhost:8501; then
    echo "ok: streamlit reachable on 8501"
  else
    echo "FAIL: streamlit didn't come up. Container logs:"
    docker logs "$CONTAINER_ID" | tail -30
    docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
    exit 1
  fi
  docker stop "$CONTAINER_ID" >/dev/null 2>&1 || true
fi

echo
echo "ALL CHECKS PASSED. Docker harness clean."
