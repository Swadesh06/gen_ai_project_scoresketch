# exp_B9_vocadito_matrix — Vocadito 2x3 (annotator x mode) baseline

## Goal
Establish a baseline for Vocadito mean COnP F1 across all combinations of annotator (A1, A2) and mode (soft, medium, hard) using the current pipeline (PESTO + voicing segmenter + B2-tuned soft defaults).

## Procedure
- Same per-clip pipeline as `gate_vocadito_conp.py`, looped through all 6 cells.
- WandB run name `gate_vocadito_matrix` with metrics nested as `{annotator}/{mode}/(mean_f1|mean_p|mean_r)`.

## Results

| | soft | medium | hard |
|---|---|---|---|
| **A1** | F1=**0.576** P=0.58 R=0.59 | F1=0.432 P=0.63 R=0.36 | F1=0.292 P=0.57 R=0.21 |
| **A2** | F1=**0.525** P=0.50 R=0.59 | F1=0.440 P=0.58 R=0.40 | F1=0.314 P=0.55 R=0.24 |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/3i240712
JSON: `reports/_gate_vocadito_matrix.json`

## Interpretation
Soft mode wins both annotators by 14-19 pp F1 over medium and 21-23 pp over hard. Medium/hard are stricter on voicing (0.50 / 0.70 vs soft's 0.32) and `min_note_seconds` (0.10 / 0.15 vs 0.05), which tanks recall for hummed material that doesn't have hard onsets or sustained pitch. Their default values are educated guesses (DESIGN_NOTES.md was explicit), and they've never been tuned for any task — so seeing them perform worse than soft on Vocadito is unsurprising.

A1 vs A2: A1 leads by ~0.05 F1 across modes, plausibly because A1 was the first annotator (potentially more careful) and the per-clip gold standard differs slightly. The cross-annotator variance gives a sense of the noise floor — improvements smaller than ~0.03 F1 should be treated as noise.

## Next
- B8: medium and hard mode sweeps. Likely the right approach is to define the modes as different *task profiles*: medium for "instrument-clean" audio (would need MAESTRO-rendered evaluation, not Vocadito); hard for "studio-clean instrument" (URMP if downloaded, or a denoised MAESTRO subset). Trying to get medium/hard to work on Vocadito (which is humming) is a category error.
- Add a `--mode auto` that picks soft for `input_kind=humming` and the current mode for instrument input.
