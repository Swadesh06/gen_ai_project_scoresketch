# exp_B87b — B87 with target_bpm=110 tempo-octave fix

## Goal
Re-run B87 (full pipeline.transcribe() on 9 ASAP pieces) with the
`humscribe.pipeline._should_use_per_voice_dp` + `target_bpm=110`
beat-tracker default. Quantify how much of B87's regression vs B63
(0.475 vs 0.774) is recoverable by fixing the tempo-octave errors that
beat_this makes on real audio.

## Results

| piece | bpm B87 | bpm B87b | snap B87 | snap B87b | snap B63 |
|---|---|---|---|---|---|
| Bach BWV 846 | 61.2 | **122.4** | 0.0321 | 0.0391 | 0.847 |
| Bach BWV 848 | 120.0 | 120.0 | 0.7154 | 0.7154 | 0.927 |
| Bach BWV 854 | 120.0 | 120.0 | 0.7667 | 0.7667 | 0.939 |
| Bach BWV 856 | 230.8 | **115.4** | 0.0357 | **0.2335** | 0.862 |
| Bach BWV 857 | 60.0 | 120.0 | 0.8046 | 0.7847 | 0.885 |
| Beethoven 21-1 | 150.0 | 150.0 | 0.6880 | 0.6880 | 0.897 |
| Schumann Toccata | 125.0 | 125.0 | 0.6111 | 0.6111 | 0.846 |
| **Chopin Berceuse** (auto-route) | 60.0 | 120.0 | **0.5690** | **0.6574** | 0.675 |
| Liszt Sonata | 115.4 | 115.4 | 0.0541 | 0.0541 | 0.053 |
| **5-Bach mean** | — | — | **0.471** | **0.508** | 0.898 |
| **4-Romantic mean** | — | — | 0.480 | **0.503** | 0.806 |
| **9-piece overall** | — | — | **0.475** | **0.506** | **0.774** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/s9tzjhq7

## Per-piece deltas (B87b − B87)

| piece | Δ snap | reason |
|---|---|---|
| BWV 846 | +0.7pp | bpm 61→122 (correct); but DP still struggles |
| BWV 848 | 0pp | tempo already correct |
| BWV 854 | 0pp | tempo already correct |
| BWV 856 | **+19.8pp** | bpm 230→115 (correct); biggest single-piece win |
| BWV 857 | -2.0pp | bpm 60→120; slight regression — duration mapping changed |
| Beethoven 21-1 | 0pp | tempo already correct |
| Schumann Toccata | 0pp | tempo already correct |
| **Chopin Berceuse** | **+8.8pp** | bpm 60→120; per_voice_dp routing also fired |
| Liszt Sonata | 0pp | tempo already correct |
| **5-Bach mean Δ** | **+3.7pp** | |
| **4-Romantic mean Δ** | +2.3pp | |
| **9-piece overall Δ** | **+3.0pp** | |

## Interpretation
- The tempo-octave fix recovers ~3pp on the 9-piece mean. Most of the
  recovery comes from BWV 856 (+20pp) where the underlying transcription
  was already accurate but the wrong-octave beats killed the duration
  matching.
- BWV 846 ALSO had its tempo doubled to 122.4 but the snap barely budged.
  This indicates that for BWV 846 specifically, the beat *positions*
  are also off (not just the BPM scaling). Even with correct doubled
  tempo, the doubled-beats midpoints don't align with note onsets.
- B87b at 0.506 is still 27pp below B63's 0.774 — the remaining gap is
  beat-positional accuracy, which would require either:
  1. A learned beat tracker fine-tuned on classical piano
  2. Score-aligned beat synthesis (less practical for real audio)

## Decision
**Keep the target_bpm=110 fix.** It's a strict improvement on average
(+3.0pp 9-piece mean, +20pp on BWV 856). No regressions worse than -2pp
on any single piece. Already committed.

The remaining beat-positional error is a Phase E target: fine-tune
beat_this on classical piano with score-aligned ground truth.

## Status
keep — tempo-octave fix integrated as `pipeline.py` default. 9-piece mean
0.475 → 0.506 (+3pp, +6.5% relative).
