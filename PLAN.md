# HumScribe v3.2 — Live Plan

Maintained by the agent.

## Status snapshot (2026-05-02 22:32)

- Phase 0 + Phase A complete; **15 Phase-B experiments** completed (3 in flight or pending).
- 17 commits ahead of origin (no remote URL provided yet — all commits local).
- WandB project `humscribe-v3.2` has 50+ runs.

## Current best metrics (vs Phase-A baselines)

| metric | Phase A | current | Δ |
|---|---|---|---|
| MIR-1K mean RPA | 0.988 | 0.988 | 0 |
| ASAP Bach BWV 846 beat-F | 0.915 | 0.915 | 0 |
| ASAP S5 snap (BWV 846, B15 voice tracking) | 0.724 | **0.779** | +5.5pp |
| ASAP mean beat-F (5 pieces, B13 octave) | n/a | **0.897** | n/a |
| ASAP mean S5 snap (5 pieces, B15) | n/a | **0.853** | n/a |
| ASAP S5 pieces ≥ 0.80 (5 pieces) | 0/5 | **4/5** | n/a |
| Vocadito A1 soft F1 (B2 sweep) | 0.538 | **0.577** | +3.9pp |
| MAESTRO instrument F1 (B14, sanity) | n/a | **0.984** | n/a |
| MTG-QBH 10-clip nonempty | 100% | 100% | 0 |

Spec target ASAP S5: 0.90 — bwv_854 hit 0.903 (first piece to clear).

## Phase B — kept improvements

1. B1 DP duration prior on offset quantization (+5.5pp ASAP raw).
2. B2 Vocadito hyperparam sweep — new soft-mode defaults.
3. B5 TPB=24 default (+2.1pp ASAP snap).
4. B13 beat_this(target_bpm=) tempo-octave correction (+6pp mean S4).
5. B14 MAESTRO instrument sanity test — pipeline saturated.
6. **B15 voice tracking + per-voice DP** (+8pp ASAP S5 snap, the largest single win).

## Phase B — discarded ideas

- B3 CREPE vs PESTO (PESTO wins 1.4pp).
- B4 default HMM (loses to voicing baseline).
- B6 HMM hyperparam sweep (HMM ceiling below voicing).
- B10 BiLSTM onset detector (small training set, sparse features).
- B11 voicing+HMM ensemble (errors correlated).

## Phase B — to try next (priority order)

1. **B16 voice tracker hyperparam sweep** (in flight) — pj × tg grid.
2. **B17 onset-DP cost prior** — penalize odd onset positions in metrical context.
3. **B18 Vocadito A2-targeted sweep** — see if A2 has different optimum (weak — A1 sweep already accommodated A2 data).
4. **B19 Onset detector with mel-spectrogram features** — re-do B10 with proper features + 5-fold CV.
5. **B20 LilyPond/MuseScore SVG rendering** — install via conda; produce real notation.

## Notes / gotchas
- TF 2.15 cuDNN-register warnings on import — cosmetic.
- ASAP MIDI rendering via fluidsynth + `pretty_midi/TimGM6mb.sf2`.
- piano_transcription_inference + beat_this both default to CUDA (auto-detect).
- mtg_qbh: `humscribe.datasets.mtg_qbh.MTGQBH` (mirdata 1.0.0 doesn't ship it).
- MAESTRO: mirdata 1.0.0 only knows v2.0.0, not 3.0.0.
- spec verbatim `eval_asap_rhythm.py` reports the index-paired metric (broken methodology, ~28%); the realistic gate is `gate_asap_rhythm.py` with VT default-on.
- `humscribe.pipeline.transcribe()` now uses voice tracking for instrument input by default.
