# exp_B6_hmm_sweep — HMM hyperparameter sweep on Vocadito

## Goal
After exp_B4 showed the default HMM segmenter underperforms the voicing-threshold segmenter (0.518 vs 0.538 mean F1), search for a hyperparameter setting where HMM beats it. Sweep all 6 HMM tuning knobs simultaneously.

## Procedure
- WandB Bayesian sweep, 24 runs, 2 parallel agents.
- Search space (`scripts/sweep_vocadito_hmm.yaml`):
  - `p_sustain ∈ uniform[0.70, 0.97]`
  - `p_end ∈ uniform[0.02, 0.20]`
  - `p_start ∈ uniform[0.02, 0.30]`
  - `sigma_voicing ∈ uniform[0.15, 0.60]`
  - `sigma_midi ∈ uniform[0.20, 1.00]`
  - `interval_decay ∈ uniform[0.30, 0.80]`
- Eval: full 40-clip Vocadito A1, soft mode, B2-tuned non-HMM hyperparameters.
- Sweep id: `agam_p-iit-roorkee/humscribe-v3.2/elxn87dj`.

## Results
Top 5 of 24 finished runs:

| F1 | P | R | p_sustain | p_end | p_start | σ_v | σ_m | int_decay |
|---|---|---|---|---|---|---|---|---|
| **0.544** | 0.659 | 0.490 | 0.878 | 0.036 | 0.094 | 0.22 | 0.97 | 0.73 |
| 0.544 | 0.647 | 0.497 | 0.904 | 0.022 | 0.085 | 0.17 | 0.92 | 0.69 |
| 0.543 | 0.646 | 0.497 | (≈ similar region) |  |  |  |  |  |
| 0.542 | 0.652 | 0.491 | (≈ similar region) |  |  |  |  |  |
| 0.541 | 0.656 | 0.486 | (≈ similar region) |  |  |  |  |  |

Comparison: voicing segmenter with B2-tuned config = **0.577** F1.

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/sweeps/elxn87dj

## Interpretation
The HMM ceiling on Vocadito (with PESTO features) is ~0.544 F1, **0.033 below** the simple voicing segmenter. Across the search, the optimum tilts toward "very sustain-heavy" (`p_sustain ~0.88-0.97`) and "rare new notes" (`p_start ~0.04-0.10`). Optimal `sigma_voicing` is small (0.17-0.26) — i.e. the model gets sharper when it trusts the voicing observation strongly.

Why HMM loses despite tuning:
1. **Voicing is the right signal here, not pitch transitions.** Vocadito hummed notes are short and use a similar pitch — the voicing trace's rising edges are highly informative for onsets, while pitch is mostly stable within voicing-defined segments. The voicing segmenter explicitly uses the rising edge; the HMM dilutes that information across all transitions.
2. **Per-frame independence in emissions** is wrong: a note onset is a local feature spread across 2-3 frames, but the HMM emission is independent per frame. Hence the natural extension: a learned onset detector with temporal context.
3. **The HMM has high precision (0.65) and low recall (0.49)** — almost the opposite trade-off of the voicing baseline. A precision-recall ensemble of the two might be worth trying.

Decision: keep voicing as default. Don't promote HMM. The implementation stays for diagnostic use.

## Next
- B10 (running): BiLSTM onset detector — directly addresses limitation (2). Trains over PESTO frame features with temporal context.
- B6c: ensemble — take HMM output as a precision-filter on voicing-segmenter output (or vice versa).
