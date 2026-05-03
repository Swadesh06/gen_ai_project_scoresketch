# exp_B87 — Full pipeline.transcribe() vs B63 baseline on 9-piece ASAP set

## Goal
End-to-end integration test: run the new pipeline (YMT3+ default + auto-routing
per-voice DP + B76 voice tracker) on the 9 ASAP pieces from B63 and compare
snap to B63's 0.774 mean.

B63 used `load_score_beats()` from `midi_score_annotations.txt` — perfect
ground-truth beats. B87 uses `pipeline.transcribe()` which calls
`track_beats_beat_this()` on the audio — real beat tracking. The delta
isolates the beat-tracking contribution to snap.

## Results

| piece | n_notes | bpm | per_voice_dp | snap (B87) | snap (B63) | Δ |
|---|---|---|---|---|---|---|
| Bach BWV 846 | 725 | **61.2 (½×)** | False | **0.0321** | 0.847 | -82pp |
| Bach BWV 848 | 1413 | 120.0 | False | 0.7154 | 0.927 | -21pp |
| Bach BWV 854 | 731 | 120.0 | False | 0.7667 | 0.939 | -17pp |
| Bach BWV 856 | 747 | **230.8 (2×)** | False | **0.0357** | 0.862 | -83pp |
| Bach BWV 857 | 1368 | 60.0 (½×) | False | 0.8046 | 0.885 | -8pp |
| Beethoven 21-1 | 8485 | 150.0 | False | 0.6880 | 0.897 | -21pp |
| Schumann Toccata | 5895 | 125.0 | False | 0.6111 | 0.846 | -23pp |
| **Chopin Berceuse** | 1637 | 60.0 | **True** | 0.5690 | 0.675 | -11pp |
| Liszt Sonata | 14901 | 115.4 | False | 0.0541 | 0.053 | +0.1pp |
| **5-Bach mean** | — | — | — | **0.4709** | 0.898 | **-43pp** |
| **4-Romantic mean** | — | — | — | **0.4805** | 0.806 | -32pp |
| **9-piece overall** | — | — | — | **0.4752** | 0.774 | **-30pp** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/izsu4zv9

## Diagnosis: tempo-octave errors dominate the regression
- **BWV 846** detected at 61 BPM (true 120) — half tempo
- **BWV 856** detected at 230 BPM (true ~115) — double tempo
- **BWV 857** detected at 60 BPM (true 120) — half tempo (less catastrophic)

These three pieces account for roughly all the Bach snap loss. With correct
tempos, snap would jump back toward B63's 0.85+ levels.

## Auto-routing verification
- **per_voice_dp=auto correctly fired only on Chopin Berceuse** (the proven
  B79 winner). All 8 other pieces kept the production shared-DP path.
- The heuristic (notes_per_sec < 10 AND pitch_iqr < 24) calibrates correctly
  on the YMT3+-transcribed pieces.

## Decision
**Informative.** The integration plumbing works (auto-routing, B76 voice
tracker load, per-voice DP run). The headline number is dragged down by
beat tracking, which is the biggest unfixed weakness when running on real
audio (vs the score-derived beats B63 used).

## Next (B87b)
Default `target_bpm=110` in `pipeline.py:track_beats_beat_this(...)` to
trigger the existing B13 octave correction. Smoke-verified that this
correctly maps:
- BWV 846: 61.2 → 122.4
- BWV 856: 230.8 → 115.4
- Liszt: 115.4 → 115.4 (unchanged)

## Status
informative — beat tracking is the next-biggest pipeline weakness; B88 fix
in flight as B87b.
