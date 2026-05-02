# HumScribe v3.2 — Live Plan

Maintained by the agent. Updated after every meaningful step.

## Status snapshot (2026-05-02)

- CPU-phase build complete: `humscribe/` package, scripts, env tarball
  (1.4 GB) at `/workspace/env-archives/humscribe.tar.gz`.
- `import humscribe` works from any cwd via a `.pth` file in the env's
  site-packages (see DESIGN_NOTES.md "Post-build fix").
- GPU now present: 1× RTX PRO 4500 Blackwell, 32 GB VRAM, driver 580 / CUDA 13.0.
- Torch installed is CPU wheel — first GPU-phase action is the swap.

## Phase 0 — unblock the GPU phase (blocking, no eval until done)

| Step | Status |
|---|---|
| 0.1 env integrity (.pth, tarball, import-from-anywhere) | done |
| 0.2 WandB online smoke (`humscribe-v3.2` project) | pending |
| 0.3 git init, .gitignore, first commit, push to GitHub SSH remote | pending |
| 0.4 torch CUDA wheel for driver 580 + repack env | pending |
| 0.5 `humscribe.datasets.mtg_qbh` Zenodo loader; patch bootstrap & eval_mtg_qbh_visual | pending |
| 0.6 `bash scripts/bootstrap.sh` in tmux; verify all 5 datasets present | pending |

## Phase A — spec gates (mandatory, in order)

| Gate | Script | Threshold | Status |
|---|---|---|---|
| Stage 2-B.1 | `scripts/eval_mir1k_pitch_sanity.py` | mean RPA > 0.85 across 5 random clips | pending |
| Stage 4 | `scripts/eval_asap_rhythm.py` | beat F-measure > 0.90 on Bach BWV 846 | pending |
| Stage 5 | `scripts/eval_asap_rhythm.py` | quarterLength match ≥ 90% | pending |
| Phase-2 visual | `scripts/eval_mtg_qbh_visual.py --modes soft,medium` | qualitative; SVGs to WandB | pending |
| Vocadito quant | new `scripts/eval_vocadito.py` | COnP F1 — pick threshold from Vocadito paper | pending |

Any failure: debug → fix → re-run → write `reports/<gate>.md` → commit → push.

## Parallelization plan

GPU has 32 GB. ByteDance piano + beat_this fit comfortably (<6 GB combined for typical clip lengths). After dry-run measurement, expect to:

- Co-schedule 2–3 independent CPU-bound eval streams while ByteDance/PESTO use the GPU.
- Use `tmux` sessions: `monitor`, `bootstrap`, `eval-<gate>`, `train-<exp>`, `sweep-<name>`.
- `nvidia-smi dmon -s pucvmet -d 5 > logs/gpu_monitor.log` runs continuously in `monitor`.

## Phase B — improvement directions (priority order)

Updated as ideas surface from papers and ablations.

1. **Pitch-tracker swap & ensemble** — CREPE-large vs PESTO on MIR-1K + Vocadito. Average their f0 estimates (median per-frame) as an ensemble. Cheap experiment, fast feedback.
2. **HMM/Viterbi note segmenter** — replace the median-filter+voicing-gated segmenter in `pitch/voicing.py` with a proper Viterbi over (pitch, voiced) states. Compare COnP F1 on Vocadito.
3. **Learned onset detector** — small TCN/Transformer trained on Vocadito + MTG-QBH, frame-level binary onset target. Replace voicing-driven onset with learned. Expect biggest gain on noisy MTG-QBH.
4. **Better DP** — swap Cemgil–Kappen for a cost that integrates note duration prior (geometric distribution over allowed quarterLengths). Tune on ASAP.
5. **YourMT3+ / MT3 end-to-end** for instrument input. Compare to ByteDance+beat_this+DP modular pipeline on MAESTRO subset.
6. **Tempo-adaptive tatum grid** — 24 tatums/beat for slow pieces, 12 for fast — pick by detected BPM.
7. **WandB hyperparameter sweep** — Bayesian search over `voicing_threshold`, `min_note_seconds`, `onset_merge_seconds`, `dp_offgrid_penalty` per mode. Multi-agent parallel.
8. **Score-rendering polish** — wire LilyPond/MuseScore so SVG is real notation, not a piano roll, for human inspection in WandB.
9. **Multi-instrument** — add saxophone/violin via Basic Pitch with input_kind=instrument, evaluate on URMP if downloadable.
10. **Self-supervised pre-training** — fine-tune a Wav2Vec2 head on MIR-1K pitch labels.

## Notes / gotchas
- TF 2.15 cuDNN-already-registered warnings on import — cosmetic, ignore.
- `mtg_qbh` not in mirdata 1.0.0 — fix in 0.5.
- ASAP needs rendered audio for some pieces; `audio_io._render_midi` falls back to FluidSynth + FluidR3_GM.sf2.
- `piano_transcription_inference` 0.0.6 first-run downloads ~330 MB lazily.
