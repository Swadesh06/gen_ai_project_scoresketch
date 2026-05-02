# exp_B14_maestro_instrument — full instrument pipeline on MAESTRO

## Goal
First quantitative test of the instrument branch (`input_kind=piano`) on MAESTRO. Five short MAESTRO Chamber clips (30 s each), full pipeline: render → ByteDance piano transcription → beat_this → DP. COnP F1 vs the score MIDI.

## Procedure
- Pick first 5 `.midi` files in `~/datasets/maestro/2018/`. Crop each to first 30 s using `pretty_midi`. Render via FluidSynth + TimGM6mb.sf2.
- For each rendered wav: run ByteDance (CUDA) → run beat_this (CUDA, no octave correction) → run DP (TPB=24).
- GT: cropped MIDI's note list (from pretty_midi).
- Metric: mir_eval `precision_recall_f1_overlap(onset_tol=0.05s, pitch_tol=50 cents, offset_ratio=None)`.
- Hardware: GPU; ~30s per clip wall-clock.

## Results

| piece | bpm | gt | pred | P | R | F1 |
|---|---|---|---|---|---|---|
| Chamber1_R3_07 | 158 | 343 | 342 | 0.988 | 0.985 | 0.987 |
| Chamber2_R3_09_1 | 65 | 200 | 203 | 0.970 | 0.985 | 0.978 |
| Chamber2_R3_09_3 | 83 | 371 | 370 | 0.978 | 0.976 | 0.977 |
| Chamber3_R3_10_1 | 77 | 134 | 136 | 0.978 | 0.993 | 0.985 |
| Chamber3_R3_10_2 | 73 | 116 | 115 | 1.000 | 0.991 | **0.996** |

- mean F1: **0.984**
- mean P: 0.983
- mean R: 0.986

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/zi7nr32u
JSON: `reports/_exp_B14_maestro_instrument.json`

## Interpretation
Nearly perfect. ByteDance was trained on MAESTRO; the rendered clips here are essentially the model's own training distribution rendered via FluidSynth (a slightly different timbre than MAESTRO's real recorded piano, but close enough that ByteDance still recovers ~99% of notes within ±50 ms and ±50 cents).

This validates the **instrument branch** end-to-end with no qualitative issues. The much lower numbers on ASAP Bach Fugues (Stage-5 ~0.74-0.80) compared to MAESTRO (~0.98) are a function of:
1. ASAP eval includes the full DP quarterLength match (much harder than just note F1).
2. ASAP's BWV 846 has 4-voice polyphonic counterpoint with overlapping notes — harder for any transcriber.
3. The MAESTRO GT here is the cropped piano performance MIDI (same source as the rendered audio), so there's no quantization mismatch.

This run is a **sanity check, not a competitive benchmark** — comparing to the published MAESTRO test-set numbers (ByteDance reports note F1 = 0.97 on the original 2018 test set) shows we're in the right neighborhood.

## Next
- B15: voice tracking — break ByteDance output into pitch-line clusters, then DP per voice. Should close the ASAP Stage-5 gap (still 23 pp shy of spec).
- Replace this dummy MAESTRO test with the real MAESTRO 2018 test split for a published-comparable number.
