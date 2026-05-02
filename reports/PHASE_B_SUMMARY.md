# Phase B — cumulative summary (2026-05-02)

## Headline numbers vs Phase A

| metric | Phase A | Phase B best | Δ | source |
|---|---|---|---|---|
| MIR-1K mean RPA | 0.988 | 0.988 | 0 | unchanged |
| ASAP BWV 846 beat-F | 0.915 | 0.915 | 0 | unchanged |
| **ASAP BWV 846 Stage-5 snap** | 0.724 | **0.847** | **+12.3pp** | B1+B5+B15+B16 |
| ASAP mean Stage-5 snap (5 pieces) | 0.773 (B12 baseline) | **0.856** | +8.3pp | B15+B16 |
| ASAP mean Stage-4 (5 pieces) | 0.836 (B12) | **0.897** | +6.1pp | B13 (eval-time only) |
| Vocadito A1 soft F1 (40 clips) | 0.538 | **0.577** | +3.9pp | B2 |
| MAESTRO instrument F1 (sanity) | n/a | 0.984 | n/a | B14 |
| MTG-QBH 10-clip nonempty | 100% | 100% | 0 | unchanged |

Bach BWV 854 reaches Stage-5 snap = **0.904** — the first piece to clear the spec target.

## What worked (kept)

1. **B1 DP duration prior** — `humscribe/rhythm/viterbi_quantize.py`: snap offset durations to the allowed musical-duration set inside the DP. Replaces the old "round to nearest tatum" which produced 7/12 quarter artefacts.
2. **B2 Vocadito hyperparam sweep** — 16-run Bayesian sweep of `voicing_threshold, min_note_seconds, pitch_smooth_window, onset_merge_seconds`. Updated `ModeConfig.for_mode("soft")` defaults.
3. **B5 default TPB=24** — `humscribe/config.py:PipelineConfig.tatums_per_beat = 24`. Reps 32nd notes exactly. Default duration set per TPB in `humscribe/rhythm/viterbi_quantize.default_allowed_durations`.
4. **B13 beat_this `target_bpm=` tempo-octave correction** — `humscribe/beat/beat_this_track.py`. Eval-time only; re-picks the predicted beats among 0.5x/1x/2x to match a known target tempo. +6pp on ASAP mean Stage-4.
5. **B15 voice tracking** — `humscribe/rhythm/voice_tracking.py`: greedy temporal+pitch-proximity voice assignment, then per-voice next-onset duration capping. **Largest single Stage-5 win (+8pp generalized).**
6. **B16 VT hyperparam sweep** — `pitch_jump=3, time_gap_s=0.5`. Tighter cluster across pieces.

## What didn't work (discarded)

- **B3 CREPE vs PESTO** — CREPE loses by 1.4pp aggregate. PESTO's voicing semantics fits the segmenter better.
- **B4/B6 HMM segmenter (default + tuned)** — structurally biased toward silence; ceiling 0.033 below voicing baseline on Vocadito.
- **B10 BiLSTM onset detector** — 30 training clips × ~30s is too small for the model size; F1 0.490.
- **B11 voicing+HMM ensemble** — errors are correlated; intersection lifts precision but kills recall.
- **B17 PESTO+CREPE per-frame ensemble** — different confidence calibrations make per-frame max picking suboptimal.

## What to try next (priority)

1. **B18 HMM voice tracker** — replace greedy with probabilistic. Should help where voice pitch lines cross.
2. **B19 mel-spectrogram BiLSTM with 5-fold CV** — proper features, more reliable val.
3. **B20 medium/hard mode sweeps** — current values are first-guess, never tuned.
4. **B21 LilyPond SVG rendering** — install via conda; produce real notation in `humscribe/score.py:render_svg`.
5. **B22 MAESTRO 2018 test set** — get a published-comparable note F1 number.

## Codebase changes summary

New modules:
- `humscribe/datasets/mtg_qbh.py` (Phase 0)
- `humscribe/pitch/hmm_segment.py` (B4/B6)
- `humscribe/pitch/ensemble.py` (B17)
- `humscribe/rhythm/voice_tracking.py` (B15/B16)
- `humscribe/train/onset_bilstm.py` (B10)

New scripts:
- 4 gate runners (mir1k, asap, mtg_qbh, vocadito) — WandB-instrumented
- 4 sweep configs/runners (vocadito × voicing/HMM, onset threshold, vt)
- 5 Phase-B experiment scripts (B11, B12, B14, B16, etc.)

Modified:
- `humscribe/config.py` — soft-mode defaults; TPB default; `note_segmenter` field
- `humscribe/pipeline.py` — voice-tracking default for instrument input
- `humscribe/beat/beat_this_track.py` — CUDA default; `target_bpm=` octave-snap
- `humscribe/instrument/piano.py` — CUDA default
- `humscribe/pitch/{pesto_track,crepe_track}.py` — CUDA default
- `humscribe/rhythm/viterbi_quantize.py` — duration prior; per-TPB allowed durations; bug-fix on candidate set
- `scripts/{bootstrap,gate_*,exp_*,sweep_*}.py` — many

## How to reproduce the headline number

```bash
conda activate humscribe
cd /workspace/swadesh/gen_ai_project_scoresketch
set -a && source .env && set +a
python scripts/gate_asap_rhythm.py
# Stage 4 0.915, Stage 5 raw 0.846, snap 0.847 — defaults are B1+B5+B15+B16.
```
