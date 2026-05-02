# exp_B1_dp_duration_prior — DP duration prior on offset quantization

## Goal
The Phase-A gate_asap_v1 result showed Stage-5 raw quarterLength match at 70% (target 90%). Inspection revealed the offset quantizer was rounding `q_off = round(off_tatum_f)` independently per note, which mapped piano-sustain noise (mean abs offset error 70 ms) onto off-grid tatum positions (e.g. 7/12-quarter durations from 16th-note inputs). Hypothesis: snapping the tatum-domain duration to a small allowed musical-duration set ({1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 48} tatums at TPB=12 = {1/12, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 1, 1.5, 2, 3, 4} quarters) inside the DP itself will lift raw match.

## Procedure
- Modified `humscribe.rhythm.viterbi_quantize._quantize_offsets`: now takes an `allowed_durations_tatums` array; for each note, computes `observed_dur = off_tatum_f - q_on`, picks the closest allowed value.
- Default for TPB=12 is `DEFAULT_ALLOWED_DURATIONS_TATUMS_TPB12`. `viterbi_quantize_rhythm` plumbs the kwarg through with the default applied when TPB=12 and no override is given. Backward compatible.
- Re-ran `scripts/gate_asap_rhythm.py` against Bach BWV 846 score-rendered audio (fluidsynth + TimGM6mb.sf2).
- Also tested an ablation B1b ("cap offset by next-onset") that hurt polyphonic Bach badly (raw 75.4 → 37.8) and was reverted.

## Results
| variant | aligned-raw | aligned-snap | verbatim |
|---|---|---|---|
| baseline (CPU build, round-only) | 0.699 | 0.724 | 0.279 |
| **B1 duration prior** (kept) | **0.754** | 0.719 | 0.289 |
| B1b add cap-by-next-onset (reverted) | 0.378 | 0.301 | 0.215 |

- WandB B1 run: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/pvbvf3ze
- WandB B1b run (failed): https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/sslhj2w6

Aligned-raw gain: +5.5 pp. Aligned-snap is statistically flat — that metric was already snapping in the eval, so the DP-side snap mostly absorbs work the eval used to do.

## Interpretation
B1 is a real win on the harder "raw" metric (pre-eval snapping). The baseline was producing nonsensical durations like 7/12 quarter that the eval-side snap then masked. With B1 the DP outputs only musically-valid durations, which is also what gets passed downstream to `score.build_stream` for MusicXML — so any user inspecting the MusicXML now sees real rhythms.

B1b (cap-by-next-onset) was tempting because polyphonic notes overlap; capping monotonically tightens long sustains. But Bach Fugue has up to 4 simultaneous voices: many "next onsets" belong to *other* voices, not the continuation of voice-i. Capping discards the legitimately long durations of held voices and writes them as ~1 tatum. Without voice tracking, this premise is broken. Logged as a Phase-B-rejected idea unless a voice-detection step is added first.

## Next
- B5 (tempo-adaptive TPB): for slow pieces like BWV 846 (48 BPM), bump TPB to 24 so 32nd notes (0.125 quarter = 3 tatums) are exactly representable instead of approximated. Should bring 32nd accuracy in line with 16th.
- Pickup B4 (HMM segmenter for humming) which doesn't depend on this — orthogonal improvement.
- Long term: voice-tracking ByteDance output before cap-by-next-onset retry.
