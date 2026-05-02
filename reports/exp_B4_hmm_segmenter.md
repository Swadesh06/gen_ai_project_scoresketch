# exp_B4_hmm_segmenter — HMM/Viterbi note segmenter (default config)

## Goal
Replace the pragmatic median-filter+voicing-thresholded note segmenter (`humscribe.pitch.voicing.segment_pitch_to_notes`) with a proper HMM/Viterbi over (silent + per-semitone) states and Gaussian midi+voicing emissions. Hypothesis: per-frame Viterbi handles voicing dropout and pitch transitions more robustly than the heuristic, lifting Vocadito F1 above the 0.538 baseline.

## Procedure
- Implementation: `humscribe.pitch.hmm_segment.segment_pitch_to_notes_hmm`. State space = 1 silent + (96-36+1)=61 active semitones (C2 through C7). Emissions: Gaussian on `(midi_obs - p)` and on `(voicing - 0)` or `(voicing - 1)`. Transitions: silent↔active jumps by `p_start=0.05`/`p_end=0.04`; active sustains by `p_sustain=0.93`; remaining mass spread over pitch jumps with geometric decay `(0.5)^|p-q|`.
- Plumbing: `PipelineConfig.note_segmenter: Literal["voicing", "hmm"]`, gate flag `--segmenter`. Default kept at `voicing` to preserve Phase-A reproducibility.
- Run: full Vocadito sweep, soft mode, A1 annotator, default HMM hyperparameters. Same harness as the Phase-A baseline.
- Performance: 0.29 s per ~33 s clip on CPU.

## Results
| segmenter | mean F1 | mean P | mean R | notes |
|---|---|---|---|---|
| voicing (baseline) | **0.538** | 0.55 | 0.55 | balanced |
| hmm (default) | 0.518 | 0.59 | 0.50 | high-precision, low-recall |

Per-clip breakdown: HMM crashes recall on a handful of clips (vocadito_3 R=0.098, vocadito_8 R=0.100, vocadito_9 R=0.097) where it predicts 4–8 notes vs 40–62 GT. On these clips the silent state dominates the Viterbi path almost entirely.

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/djqf86nn

## Interpretation
The HMM is statistically too conservative with the default config. Several explanations to investigate:
1. **PESTO voicing calibration** — our emission model assumes voicing ∈ {≈0 silent, ≈1 voiced}. PESTO's `conf` might cluster differently (e.g. ~0.3-0.7 as the "voicing band"), making the silent emission model better-fitting on average.
2. **`sigma_voicing=0.30` is too tight** for the actual voicing-confidence distribution; raising it should let voiced frames register more strongly.
3. **`p_start=0.05` is low** — note onsets are rare relative to sustained frames, but with 100 Hz frame rate and short hummed notes this ratio is too small.

Not a usable replacement out of the box. Needs hyperparameter tuning (or, better, EM on a held-out set).

## Next
- B4b: HMM hyperparameter sweep — `p_start, p_end, p_sustain, sigma_voicing, sigma_midi, interval_decay`. Bayesian over 30 runs. Defer until after B2 (so we know what voicing-segmenter best looks like; HMM should beat that).
- Diagnose the conf distribution: histogram PESTO confidence on Vocadito, fit a 2-component Gaussian mixture; use those as emission priors.
- B4c: try CREPE-as-pitch with the HMM (CREPE periodicity might be better-calibrated than PESTO confidence).
