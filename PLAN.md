# HumScribe v3.2 — Live Plan

Maintained by the agent.

## Status snapshot (2026-05-02 22:20)

- Phase 0 complete; Phase A complete; **13 Phase-B experiments completed**.
- 16 commits ahead of origin (no remote URL provided yet).
- WandB project `humscribe-v3.2` has 30+ runs across baselines, sweeps, ablations.

## Current best metrics

| metric | Phase-A baseline | current best | source |
|---|---|---|---|
| MIR-1K mean RPA (5 clips) | 0.988 | 0.988 | gate_mir1k |
| ASAP Bach BWV 846 beat-F | 0.915 | 0.915 | (octave correction is no-op here) |
| ASAP Stage-5 snap (BWV 846) | 0.724 | **0.740** | B1 + B5 |
| ASAP mean beat-F (5 pieces) | n/a | **0.897** | B13 |
| ASAP mean Stage-5 snap (5 pieces) | n/a | **0.773** | B12 |
| Vocadito A1 soft F1 (40 clips) | 0.538 | **0.577** | B2 sweep |
| Vocadito A2 soft F1 | n/a | 0.525 | B9 |
| MTG-QBH visual nonempty | 20/20 | 20/20 | unchanged |

## Phase B — improvement loop

### Done (and either kept or discarded)
- B1 DP duration prior — keep, +5.5pp ASAP raw
- B2 Vocadito sweep — keep, +3.9pp F1
- B3 CREPE vs PESTO — PESTO wins
- B4 HMM segmenter (default) — discard
- B5 TPB=24 default — keep, +2.1pp
- B6 HMM hyperparam sweep — discard, ceiling 0.033 below voicing
- B7 MTG-QBH re-baseline — keep
- B9 Vocadito 2x3 matrix — keep as baseline
- B10 BiLSTM onset detector — discard, needs more data
- B11 voicing+HMM ensemble — discard, errors correlated
- B12 ASAP multi-piece — keep
- B13 tempo-octave correction — keep, +6pp mean Stage-4

### To try next (priority order)
1. **B14 MAESTRO instrument test** — full pipeline on 5 short MAESTRO clips with input_kind=piano. First quantitative test of medium/hard modes for instrument input.
2. **B15 voice tracking** — cluster ByteDance notes by pitch line, quantize per-voice. The 23pp gap from Stage-5 to spec target is mostly polyphonic confusion.
3. **B16 onset detector with mel-spectrogram** — re-do B10 with proper features (32-band log-mel) and 5-fold CV for reliable val numbers.
4. **B17 SwiftF0 alternative** — license-clean PESTO replacement.
5. **B18 MAESTRO sweep** — sweep medium/hard mode hyperparameters against MAESTRO-rendered audio.

## Parallelization

GPU at 32 GB. Most experiments use <3 GB; can run 2-3 in parallel. Sweeps run 2 agents safely.

## Notes / gotchas
- TF 2.15 cuDNN-already-registered warnings on import — cosmetic, ignore.
- ASAP MIDI rendering via fluidsynth + `pretty_midi/TimGM6mb.sf2`.
- piano_transcription_inference now defaults to CUDA via `_autodevice`.
- beat_this now defaults to CUDA + supports `target_bpm=` for evaluation.
- mtg_qbh: not in mirdata; use `humscribe.datasets.mtg_qbh.MTGQBH`.
- MAESTRO: mirdata 1.0.0 only knows v2.0.0, not 3.0.0.
- spec verbatim `eval_asap_rhythm.py` reports the index-paired metric; the realistic gate is `gate_asap_rhythm.py`.
