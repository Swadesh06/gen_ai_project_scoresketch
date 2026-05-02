# exp_B17_pitch_ensemble — PESTO+CREPE per-frame max-confidence ensemble

## Goal
B3 showed CREPE alone loses to PESTO by 1.4pp on Vocadito A1 aggregate, but wins on a few specific clips (e.g. vocadito_8 +0.23 F1). Per-frame ensemble that picks the more-confident tracker per frame might combine both strengths.

## Procedure
- `humscribe.pitch.ensemble.track_pitch_ensemble`: run both, interpolate CREPE to PESTO frame grid, output `(pesto_t, hz_argmax_conf, max_conf)`.
- `gate_vocadito_conp.py --pitch-model ensemble`. All 40 Vocadito A1 clips.

## Results
| pitch model | mean F1 |
|---|---|
| PESTO (default) | 0.576 |
| CREPE | 0.562 |
| **Ensemble (B17)** | 0.553 |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/ri8vielq

## Interpretation
Worse than either tracker alone — by 2.3pp vs PESTO. The reason mirrors B11's ensemble failure: CREPE periodicity has a different statistical scale than PESTO confidence, so "max-conf per frame" frequently picks CREPE when PESTO was better-calibrated for that voicing strength. Calibration would need a shared confidence space (e.g., via temperature-scaling on a held-out set) — overfitting risk on Vocadito's small size.

Discard. PESTO remains the default.

## Next
- Skip pitch-tracker ensembling. Move to a different idea (HMM voice tracker for ASAP).
