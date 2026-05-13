# item-g17 — Docker actual-build verification artifact

## Goal
task_description_v4.md item G-17. Close v3 spec item 4 ("`docker build` succeeds") by shipping a build + smoke harness the user can run on a real Linux host. The agent does NOT run `docker build` (the sandbox has no docker binary).

## Procedure
- New script `scripts/g17_docker_verify.sh` — single bash file, exits non-zero on any failure. Checks in order:
  1. `docker` on PATH
  2. docker daemon reachable (`docker info`)
  3. required files: `Dockerfile`, `.dockerignore`, `requirements.txt`; required dirs: `humscribe/`, `app/`, `scripts/`, `third_party/`
  4. `docker build -t humscribe:phase-g .`
  5. import smoke: `docker run --rm humscribe:phase-g python -c "import humscribe; from humscribe.eval.mv2h import compute_mv2h; print('ok')"`
  6. optional streamlit smoke: detach a container on 8501, curl localhost, stop. Skipped with `--no-run`.
- Uses the existing Phase E Dockerfile + .dockerignore unchanged. `g17_docker_verify.sh` is the new artifact that turns the Dockerfile from "validates statically" (v3 strict-fail status) into "runs to completion on a docker host".

## Results
- Script shipped at `scripts/g17_docker_verify.sh`, marked executable.
- In-sandbox check: docker binary not present (expected — see CLAUDE.md note "sandbox has no docker binary, can't run").
- User-side run produces a clear PASS/FAIL with the failing step named.

## Pass / discard
- **Script shipped + executable**: PASS.
- **Script exits 0 on a real docker host**: → **deferred to the user's host** (the agent doesn't run docker).

**Net G-17 status: HUMAN-FACING ARTIFACT SHIPPED. The strict v3 item 4 criterion (`docker build` succeeds) becomes a one-command verification — `bash scripts/g17_docker_verify.sh` — for the user.**

## Next
After the user runs the script: if it fails at step 4 (build), debug the failure and update the Dockerfile. If it passes through step 6, the v3 item 4 criterion is closed in the production sense.
