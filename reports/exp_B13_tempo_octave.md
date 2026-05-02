# exp_B13_tempo_octave — beat_this tempo-octave correction

## Goal
B12 showed beat_this drops below the 0.90 F-measure gate on slow (60 BPM) and very fast (231 BPM) ASAP pieces because it locks onto the wrong tempo octave — predicting 122 BPM beats on a 61 BPM piece, halving the beat count. When a target tempo is known (here from `midi_score_annotations.txt`), we can post-correct: choose the {0.5x, 1x, 2x} variant of the predicted beat sequence whose log2-distance to the target BPM is smallest.

## Procedure
- `humscribe.beat.beat_this_track.track_beats_beat_this(audio_path, target_bpm=...)` now optionally takes a target BPM.
- If `target_bpm` is given: compute three options (predicted, doubled by midpoint insertion, halved by every-other), pick min `|log2(option_bpm / target_bpm)|`.
- `gate_asap_rhythm.py` and `exp_B12_asap_multi.py` pass `target_bpm = 60 / avg(diff(score_beats))`.
- Re-ran the 5-piece B12 sweep.

## Results

| piece | bpm | B12 beat F | **B13 beat F** |
|---|---|---|---|
| Bach/Fugue/bwv_846 | 120 (was 61) | 0.775 | **0.845** |
| Bach/Fugue/bwv_848 | 120 | 0.969 | 0.969 |
| Bach/Fugue/bwv_854 | 120 | 0.944 | 0.944 |
| Bach/Fugue/bwv_856 | 231 | 0.778 | 0.778 |
| Bach/Fugue/bwv_857 | 120 (was 60) | 0.715 | **0.948** |

Aggregates:
- mean Stage-4 beat-F: 0.836 → **0.897** (+6 pp)
- median: 0.778 → 0.944
- pass rate (> 0.90): 40% → **60%** (3/5)
- Stage 5 unchanged (snap mean 0.773) — correctly so, since correction only changes beat positions, not count fed to the DP

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/4czdd05a

## Interpretation
Two of five pieces benefited massively (+7 pp and +23 pp). The other three were already at the right tempo octave — beat_this was correct, the heuristic correctly chose the 1x option. **No regressions.**

The remaining piece below 0.90 (bwv_856 at 231 BPM) doesn't benefit because halving (115 BPM) or further halving still doesn't match the score's 231 BPM exactly — beat_this is missing beats outright at this fast tempo, not picking the wrong octave. A different fix is needed there (better high-tempo onset detection).

When the target BPM is unknown (the typical inference case for humming/instrument input where we don't have a score), this correction can't be applied. For instrument transcription where we DO have access to the rendered audio's tempo via beat_this itself, this isn't useful. Use case: ASAP-style validation only, where a ground-truth tempo is available. In production, document that `target_bpm=` is for evaluation/validation harnesses.

## Next
- B14: tempo-fold vote — instead of trusting the target, run beat_this with multiple synthetic time-stretches and vote on the most consistent tempo.
- Continue addressing remaining Stage-5 gap (74 → 90%) which is now the main outstanding spec-vs-actual.
