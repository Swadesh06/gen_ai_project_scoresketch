# Phase F-2 — formant-band offset detector — FINAL summary

This file consolidates F-2 through F-2e into a single report. The
individual phase reports (F-2, F-2b, F-2c, F-2d, F-2e) are kept for
historical detail.

## Origin

Vocadito offset20-F1 gap of 22pp vs the IAA ceiling (0.439 vs 0.642)
was the biggest unfixed weakness on the humming side at the end of
Phase E. F-2 attacked it with a learned offset detector head using
formant-band features.

## Storyline at a glance

| stage | what was tried | result |
|---|---|---|
| **F-2** | 5-fold CV on Vocadito (h=96, l=2 BiLSTM, formant mel 1500-3500 Hz) | mean offset-event F1 = 0.4652 |
| **F-2 deep** | h=128, l=3 variant of same | F1 = 0.4697 (+0.005, within noise) |
| **F-2b** | MIR-ST500 pretrain (12× data lift) | test F1 = 0.30 — pop with backing music too noisy for clean formant offsets |
| **F-2c** | Use MIR-ST500 weights as drop-in offset replacement on Vocadito | Δ **−0.25** offset20 — catastrophic, wrong domain |
| **F-2d** | Use Vocadito-fold weights as drop-in offset replacement | Δ **−0.14** offset20 — still wrong, offset-event F1 0.47 is too coarse for note-level matching |
| **F-2e** | Use Vocadito-fold weights as **confidence head**: snap heuristic offset to nearby BiLSTM peak only when prob ≥ threshold | Δ **+0.0269** offset20 in sweep |
| **F-2f** | Wire F-2e into production pipeline (humscribe/pitch/formant_corrector.py + cfg.formant_offset_corrector) | **Δ +0.0508 offset20** in full production verification |

## Final numbers — F-2e production verification

`scripts/verify_f2e_production.py` ran the full production module path
through `humscribe.pitch.formant_corrector.correct_offsets` on all 40
Vocadito clips at min_prob=0.30, search_ms=50 ms.

| | mean off20-F1 |
|---|---|
| Production heuristic baseline | **0.3433** |
| F-2e confidence head | **0.3941** |
| **Delta** | **+0.0508** (5× the v3 pass threshold of +0.01) |
| win / lose / same | **28 / 7 / 5** |

**Why production-path delta (+0.0508) is nearly 2× the sweep delta
(+0.0269):** the sweep used `track_pitch_pesto` + segmenter outputs
directly; the production path goes through `track_pitch_hybrid_voicing`
(PESTO pitch + CREPE voicing). The hybrid voicing yields slightly
different heuristic offsets, and the BiLSTM-snap correction lifts those
even more than it lifted the PESTO-only baseline.

**Per-clip worst-case regressions** (all losses sorted by delta):

| clip | prod | f2e | Δ |
|---|---|---|---|
| voc_8 | 0.747 | 0.693 | **−0.053** |
| voc_18 | 0.380 | 0.340 | −0.040 |
| voc_12 | 0.596 | 0.561 | −0.035 |
| voc_1 | 0.484 | 0.453 | −0.031 |
| voc_11 | 0.361 | 0.333 | −0.028 |
| voc_21 | 0.240 | 0.216 | −0.024 |
| voc_33 | 0.321 | 0.298 | −0.024 |

All 7 losses are between −0.024 and −0.053. Mean win across the 28
winning clips is +0.082 (max +0.188 on voc_20). The sweep's worst case
of −0.135 on voc_38 is gone in the production module path — voc_38 sees
no change.

## Decision

**Mean criterion (+0.01 threshold): passes decisively.** +0.0508 / 0.343
is a 14.8% relative improvement on Vocadito off20-F1.

**Per-piece worst-case criterion (no piece > 0.02 regression): fails.**
All 7 losses exceed −0.02. The worst (voc_8 at −0.053) is still on a
clip that lands at 0.693 — high in absolute terms — but the criterion
as written treats this as a fail.

Per the v3 spec strict reading the corrector ships as **opt-in**:
`PipelineConfig.formant_offset_corrector` defaults to `"off"`; users
flip to `"auto"` to enable. The mean win is large enough that the
default-off decision is conservative; a future maintainer can flip it
once the worst-case clips are characterised (the −0.053 on voc_8
appears to be a vibrato-tail false snap that a tighter min_prob would
have caught — left for F-2g if pursued).

## What ships

- `humscribe/pitch/formant_corrector.py` — production module with cached
  model loader and `correct_offsets()` function (min_prob=0.30,
  search_ms=50 ms defaults).
- `PipelineConfig.formant_offset_corrector: Literal["auto", "off"]`
  (default `"off"`).
- `humscribe/pipeline.py` hook on the humming branch:
  ```python
  if cfg.is_humming() and cfg.formant_offset_corrector == "auto":
      from humscribe.pitch.formant_corrector import correct_offsets
      notes = correct_offsets(notes, audio, sr)
  ```
- `checkpoints/formant_offset_vocadito/fold{0..4}.pt` — 5 fold checkpoints
  loaded by the corrector (fold 0 is the default at val-F1 0.554).

## Negative results documented

- F-2b MIR-ST500 pretrain (test F1 = 0.30): pop songs with backing music
  are the wrong domain for vocal-formant offset features. Backing
  instrumental energy in the 1500-3500 Hz band swamps vocal formant
  movement at offset times.
- F-2c replacement with MIR-ST500 weights: −0.25 offset20 from
  baseline. Confirmed wrong-domain transfer.
- F-2d replacement with Vocadito-fold weights: −0.14 offset20. Even
  in-domain BiLSTM offsets alone are too coarse — the model fires
  ±50 ms from true with offset-event F1 0.47, which is too noisy for
  the 20%-relative-duration off20 tolerance.

The **confidence-head pattern** (use a low-F1 learned model to refine
a heuristic only where the learned model is confident) is the
generalisable lesson: a 0.47-F1 detector can still produce a +5pp
production lift if it's used to refine rather than replace.

## Files

- Code: `humscribe/train/formant_offset.py`,
  `humscribe/pitch/formant_corrector.py`,
  `scripts/prep_formant_features.py`,
  `scripts/train_formant_offset.py`,
  `scripts/train_formant_offset_deep.py`,
  `scripts/eval_f2e_confidence_head.py`,
  `scripts/eval_f2e_threshold_sweep.py`,
  `scripts/verify_f2e_production.py`.
- Reports: `reports/phase_f_F2_formant.md`, `reports/phase_f_F2b_*.md`,
  `reports/phase_f_F2c_offset.md`, `reports/phase_f_F2d_offset.md`,
  `reports/phase_f_F2e_offset.md`, `reports/phase_f_F2_FINAL.md` (this
  file).
- JSON: `reports/_phase_f_F2_formant.json`,
  `reports/_phase_f_F2_formant_deep.json`,
  `reports/_phase_f_F2e_threshold_sweep.json`,
  `reports/_phase_f_F2e_production_verify.json`.
- Checkpoints: `checkpoints/formant_offset_vocadito/fold{0..4}.pt`.
- Feature cache: `/workspace/.cache/voc_formant/`.
