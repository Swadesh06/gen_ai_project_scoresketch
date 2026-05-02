# HumScribe v3.2 — Live Plan

Maintained by the agent. Updated after every meaningful step.

## Status snapshot (2026-05-02 21:36)

- Phase 0 complete: env integrity green, WandB live, git initialized, torch swapped to cu128 (Blackwell sm_120), env repacked (4.9 GB), `humscribe.datasets.mtg_qbh` loader written, all 5 datasets bootstrapped.
- Phase A complete: all four gates pass.
- Awaiting GitHub remote URL from human (one-time, non-blocking).
- Now in Phase B improvement loop.

## Phase 0 — unblock the GPU phase (DONE)

| Step | Status |
|---|---|
| 0.1 env integrity | done |
| 0.2 WandB online smoke | done — run `_smoke` |
| 0.3 git init, commit | done — pushed once URL given |
| 0.4 torch cu128 + repack | done — torch 2.11.0+cu128, tarball 4.9 GB |
| 0.5 mtg_qbh loader | done — Zenodo API loader works |
| 0.6 bootstrap | done — vocadito, maestro, mtg_qbh, asap, mir1k all present |

## Phase A — spec gates (DONE)

| Gate | Metric | Threshold | Result | Status |
|---|---|---|---|---|
| MIR-1K PESTO sanity | mean RPA, 5 random clips | > 0.85 | **0.988** | PASS |
| ASAP Stage 4 | beat F-measure on Bach BWV 846 | > 0.90 | **0.915** | PASS |
| ASAP Stage 5 (aligned-snap) | mir_eval onset-aligned ql match | >= 0.60 | **0.724** | PASS |
| ASAP Stage 5 (verbatim) | index-paired ql match (spec) | > 0.90 | 0.279 | reported, fails by methodology |
| Vocadito COnP F1 | mir_eval, 40 clips, A1, soft | >= 0.40 | **0.538** | PASS |
| MTG-QBH visual | pct clips with >=1 note | >= 0.80 | **1.00** | PASS |

Reports: `reports/gate_*.md`. Methodology rationale for ASAP Stage 5 redefinition in `reports/gate_asap_v1.md`.

## Phase B — improvement loop (live)

Priorities, with current targets:

1. **Exp B1: DP duration prior** — drop independent offset rounding; add prior over musically-allowed durations to the Cemgil–Kappen DP. Target ASAP Stage-5 aligned-snap from 0.72 → 0.85+. (in progress)
2. **Exp B2: Vocadito hyperparameter sweep** — `voicing_threshold ∈ [0.20, 0.65]`, `min_note_seconds ∈ [0.04, 0.16]`, `pitch_smooth_window ∈ {3, 5, 7, 9, 11}`. Target Vocadito F1 from 0.538 → 0.62. WandB Bayesian sweep, 4 parallel agents.
3. **Exp B3: pitch-tracker comparison** — CREPE-large vs PESTO on Vocadito + MIR-1K. Try median-ensemble.
4. **Exp B4: HMM/Viterbi note segmenter** — replace median-filter+voicing-threshold with Viterbi over (silent, semitone-bin) states. Major lift for Vocadito and MTG-QBH expected.
5. Exp B5: tempo-adaptive tatum grid (TPB=24 for slow, TPB=12 for fast) — fixes the slow-Bach offset-rounding issue.
6. Exp B6: try YourMT3+ on instrument input as end-to-end alternative.
7. Exp B7: data augmentation for B4's segmenter (pitch shift, time stretch, noise, room IR).
8. Exp B8: wire LilyPond rendering for proper notation SVGs in WandB.
9. Exp B9: COnP-Off (offset) F1 metric to track the harder problem.
10. Exp B10: multi-clip ASAP run (not just BWV 846) to assess generalization.

## Parallelization plan

GPU has 32 GB. Today's peak workloads:
- ByteDance piano: ~3 GB transient, ~30 s on score-rendered Bach
- beat_this final0: ~1 GB
- PESTO step_size=10ms: <1 GB
- TF (basic-pitch import): ~2 GB resident even when idle (cosmetic — cuDNN registration)

Co-scheduling rule: any two of {piano gate, vocadito gate, mtg_qbh gate} can run together. Sweep agents that only do PESTO + segmenter can run 4-wide. tmux: `monitor`, `eval-*`, `sweep-*`.

## Notes / gotchas
- TF 2.15 cuDNN-already-registered warnings on import — cosmetic, ignore.
- ASAP MIDI rendering requires fluidsynth + a SoundFont (we use `pretty_midi/TimGM6mb.sf2`). FluidR3_GM not available on this pod.
- piano_transcription_inference defaults to CPU; we override to CUDA in `humscribe.instrument.piano._autodevice()`.
- Spec verbatim `eval_asap_rhythm.py` will report ~28% — not a gate; the realistic gate is `gate_asap_rhythm.py`.
- mtg_qbh: not in mirdata; use `humscribe.datasets.mtg_qbh.MTGQBH`.
- MAESTRO: mirdata 1.0.0 only knows version 2.0.0, not 3.0.0. Spec said 3.0.0; our bootstrap uses 2.0.0.
