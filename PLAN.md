# HumScribe v3.2 — Live Plan

Maintained by the agent.

## Status snapshot (2026-05-02 22:42)

- Phase 0 + Phase A complete; **17 Phase-B experiments** done (7 keep, 7 discard, 3 informative).
- 23 commits ahead of origin (no remote URL provided yet — all commits local).
- WandB project `humscribe-v3.2` has 60+ runs.

## Current best metrics vs Phase A baselines

| metric | Phase A | current | Δ | source |
|---|---|---|---|---|
| MIR-1K mean RPA | 0.988 | 0.988 | 0 | unchanged |
| ASAP BWV 846 beat-F | 0.915 | 0.915 | 0 | unchanged |
| **ASAP BWV 846 Stage 5 snap** | 0.724 | **0.847** | **+12.3pp** | B1+B5+B15+B16 |
| ASAP BWV 846 Stage 5 raw | 0.699 | 0.846 | +14.7pp | same |
| ASAP mean Stage-4 (5 pieces) | n/a | 0.897 | n/a | B13 |
| **ASAP mean Stage-5 snap (5 pieces)** | n/a | **0.856** | n/a | B15+B16 |
| Vocadito A1 soft F1 (40 clips) | 0.538 | **0.577** | +3.9pp | B2 |
| MAESTRO instrument F1 (sanity) | n/a | 0.984 | n/a | B14 |
| MTG-QBH 10-clip nonempty | 100% | 100% | 0 | unchanged |

Spec target ASAP S5: 0.90. Bach BWV 854 is at 0.904 (one piece passes the spec target).

## Phase B — kept improvements

1. B1 DP duration prior on offset quantization
2. B2 Vocadito hyperparam sweep — new soft-mode defaults (vt=0.315, mns=0.052, psw=11, oms=0.026)
3. B5 TPB=24 default (32nd-note exact representation)
4. B13 beat_this(target_bpm=) tempo-octave correction (eval-time only)
5. B14 MAESTRO instrument sanity test — pipeline saturated
6. **B15 voice tracking + per-voice DP** (largest single Stage-5 win)
7. **B16 VT hyperparam sweep** (pitch_jump=3, time_gap_s=0.5)

## Phase B — discarded
- B3 CREPE vs PESTO — PESTO wins
- B4 default HMM segmenter
- B6 HMM hyperparam sweep
- B10 BiLSTM onset detector
- B11 voicing+HMM ensemble
- B17 PESTO+CREPE per-frame max-conf ensemble

## Phase B — to try next

1. **B18 HMM voice tracker** — true probabilistic voice assignment, beats greedy on crossing pitch lines.
2. **B19 BiLSTM with mel-spectrogram features + 5-fold CV** — re-do B10 with proper inputs.
3. **B20 Medium/hard mode sweeps** — they were never tuned for any task.
4. **B21 LilyPond/MuseScore SVG** — proper notation in WandB.
5. **B22 Real MAESTRO test set + published-comparable F1** — only sanity-tested so far.

## Notes / gotchas
- TF 2.15 cuDNN-register warnings on import — cosmetic.
- ASAP MIDI rendered via fluidsynth + pretty_midi/TimGM6mb.sf2.
- piano_transcription_inference + beat_this default to CUDA (auto-detect).
- mtg_qbh: humscribe.datasets.mtg_qbh.MTGQBH (mirdata 1.0.0 lacks it).
- MAESTRO: mirdata 1.0.0 only knows v2.0.0.
- spec verbatim eval_asap_rhythm.py reports the index-paired metric (~28%, broken methodology); realistic gate is gate_asap_rhythm.py with VT default-on.
- humscribe.pipeline.transcribe() now uses voice tracking for instrument input by default.
