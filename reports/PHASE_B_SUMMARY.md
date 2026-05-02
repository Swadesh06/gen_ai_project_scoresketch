# Phase B — cumulative summary (final, 2026-05-02)

## Final headline metrics vs Phase A baselines

| metric | Phase A | Phase B best | Δ | source |
|---|---|---|---|---|
| MIR-1K mean RPA | 0.988 | 0.988 | 0 | unchanged (already saturated) |
| ASAP BWV 846 beat-F | 0.915 | 0.915 | 0 | unchanged |
| **ASAP BWV 846 Stage-5 snap** | 0.724 | **0.847** | **+12.3pp** | B1+B5+B15+B16 |
| ASAP mean Stage-5 snap (5 pieces) | n/a | **0.856** | n/a | B12+B15+B16 |
| ASAP mean Stage-4 (5 pieces) | 0.836 | **0.897** | +6.1pp | B13 |
| **Vocadito A1 soft F1 (40 clips)** | 0.538 | **0.597** | **+5.9pp** | B2 + B22 |
| Vocadito A2 soft F1 | 0.525 | 0.551 | +2.6pp | B22 |
| MAESTRO instrument F1 (sanity) | n/a | 0.984 | n/a | B14 |
| MTG-QBH 10-clip nonempty | 100% | 100% | 0 | unchanged |

Bach BWV 854 reaches Stage-5 snap = **0.904** — first piece to clear the spec target.

## Phase B count: 23 experiments, 8 keep, 9 discard, 6 informative

### Kept (production)
1. **B1** DP duration prior on offset (snap-to-allowed-tatum) — `humscribe/rhythm/viterbi_quantize.py`
2. **B2** Vocadito hyperparam sweep — soft-mode defaults (vt=0.315, mns=0.052, oms=0.026)
3. **B5** default TPB=24 — `humscribe/config.py:PipelineConfig.tatums_per_beat`
4. **B13** beat_this `target_bpm=` tempo-octave correction — `humscribe/beat/beat_this_track.py`
5. **B14** MAESTRO instrument sanity test (validation) — pipeline at 0.984 F1
6. **B15** voice tracking + per-voice DP — `humscribe/rhythm/voice_tracking.py`
7. **B16** voice-tracker hyperparams (pj=3, tg=0.5)
8. **B18** Verovio real-notation SVG rendering — `humscribe/score.py:_verovio_svg`
9. **B22** Vocadito psw=15 (extended sweep range) — soft-mode pitch_smooth_window

### Discarded
- B3 CREPE (loses to PESTO 1.4pp aggregate)
- B4 default HMM segmenter (worse than voicing)
- B6 HMM hyperparam sweep (ceiling 0.033 below voicing)
- B10 BiLSTM onset detector (small data)
- B11 voicing+HMM ensemble (errors correlated)
- B17 PESTO+CREPE per-frame max-conf ensemble (calibration mismatch)
- B19 mel-BiLSTM 5-fold CV (still data-limited)
- B20/B21/B25 HMM voice tracker (loses to greedy on Bach)
- B23 DP hyperparam sweep on BWV 854 (already at optimum)
- B27 CREPE with psw=15 (still loses to PESTO+psw=15)
- B33 dense PESTO step_ms (5ms hurts; 15ms within noise)
- B34 basic_pitch on Vocadito (too-many-notes, F1 0.495)
- B35 librosa onset_detect (F1 0.380)

### Informative
- B7 MTG-QBH re-baseline after B2 (still 100% nonempty)
- B9 Vocadito A1×A2 × soft/medium/hard 2x3 matrix
- B12 ASAP multi-piece B1+B5 wins generalize
- B26 vt/mns/oms re-sweep with psw=15 (no further gain — ceiling 0.597)
- B28 A2 with psw=15 (+2.6pp generalization confirmed)

## Cumulative Vocadito A1 trajectory
- Phase A baseline: 0.538
- After B2 sweep: 0.577 (+3.9pp)
- After B22 psw=15: 0.597 (+2.0pp)
- **Total: +5.9pp**

The ceiling for the PESTO+voicing-segmenter pipeline on Vocadito is **0.597**.

## Cumulative ASAP BWV 846 Stage-5 snap trajectory
- Phase A baseline: 0.724
- After B1 (duration prior): 0.719 (within noise)
- After B5 (TPB=24): 0.740
- After B15 (voice tracking, default): 0.779
- After B16 (VT sweep tuning): 0.847
- **Total: +12.3pp**

ASAP multi-piece mean: 0.773 → 0.856 (+8.3pp).

## What's left (Phase B+1 ideas, not yet attempted)

1. **Train an onset detector with much more data** (Vocadito + MIR-1K voicing + synthesised augmented humming) — only B10/B19 attempted in-distribution.
2. **HMM voice tracker with better proposals** — current beam-search greedy assignment gets stuck on default config. A learned voice assigner might push ASAP S5 from 0.85 → 0.90.
3. **Tempo-modeled Stage-4 fix for fast pieces** — bwv_856 (231 BPM) is the laggard at S4=0.78 and S5=0.81.
4. **MAESTRO 2018 test split with audio** — currently only sanity-tested with rendered MIDI; would give published-comparable note F1.
5. **MERT/MusicFM features** as input to learned segmenter — pre-trained music encoder.

## Codebase final state (in addition to Phase A skeleton)
- `humscribe/datasets/mtg_qbh.py` — Zenodo loader
- `humscribe/pitch/{hmm_segment, ensemble}.py` — discarded but retained for future work
- `humscribe/rhythm/{voice_tracking, voice_hmm}.py` — VT modules (greedy is default)
- `humscribe/train/{onset_bilstm, onset_mel}.py` — discarded BiLSTM modules
- `humscribe/score.py` — Verovio renderer wired in (B18)
- `humscribe/beat/beat_this_track.py` — `target_bpm=` octave-snap added
- `humscribe/config.py` — defaults updated 4 times (B2, B5, B16 for VT, B22)

40+ commits, 32+ scripts, 23+ reports. WandB project `humscribe-v3.2` at https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2.
