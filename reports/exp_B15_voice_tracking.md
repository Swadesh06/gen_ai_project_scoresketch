# exp_B15_voice_tracking — voice tracking + per-voice DP

## Goal
ASAP Stage 5 has been stuck around 74% snap (single piece) / 77% mean (5 pieces) — well below the spec's 90% target. The remaining error is mostly polyphonic-overlap noise: ByteDance's per-note offset detection bleeds across voice boundaries (a held bass note is detected as ending only when its acoustic energy drops, which is often after the next soprano note onset). For a per-voice rhythm, the right "duration" is the gap to the next note **in the same voice**, which the existing DP cannot infer because it sees a flat time-ordered note list.

Fix: greedy voice-tracking pass before DP. Each note is attached to the most recent voice whose last pitch is within `pitch_jump=4` semitones and whose offset is within `time_gap_s=1.5` of the new onset; otherwise a new voice starts. Then per-voice "duration = time to next onset in same voice" caps the original offset, so the DP sees clean monophonic timing per voice.

## Procedure
- Implementation: `humscribe.rhythm.voice_tracking.{assign_voices, per_voice_durations, quantize_with_voice_tracking}`.
- Activated via `gate_asap_rhythm.py --voice-tracking`. Tested on the same Bach BWV 846 score-rendered audio as the Phase-A baseline, using B1+B5 defaults (TPB=24, allowed-duration snap).
- Multi-piece test (B15-multi) wires the same path through `exp_B12_asap_multi.py`.

## Results — Bach BWV 846

| variant | aligned-raw | aligned-snap |
|---|---|---|
| B1+B5 baseline (no voice tracking) | 0.754 | 0.719 |
| **B15 voice tracking** | **0.837** | **0.779** |
| Δ | **+8.3pp** | **+6.0pp** |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/rt6rj7g7

## Interpretation
Voice tracking is the largest single Stage-5 lift in Phase B. The improvement on `aligned-raw` (+8.3 pp) is bigger than on `aligned-snap` (+6.0 pp) because raw measures exact-quarter match while snapped already accepts some jitter — voice tracking primarily fixes the "duration is way off" cases (e.g., 7/12 quarters → 6/12 quarters) where snap was already collapsing similar offsets.

The Bach Fugue is 4-voice — voice tracking should help it disproportionately. We'd expect smaller gains on monophonic pieces. Multi-piece test (B15-multi) in flight will quantify this; if mean lift is similar across the 5 Bach Fugues (also 4-voice), it generalizes; if it's flat for less polyphonic pieces, the gain is voice-density-dependent.

The greedy voice tracker is intentionally simple. Failure modes likely include: (a) wrong voice assignment when two voices cross in pitch, (b) fragmenting a single voice into many when the singer/player jumps. Both are addressable by a smarter assigner (e.g., HMM over voices), Phase-B priority "next".

## Results — multi-piece (5 Bach Fugues)

| piece | bpm | beat F | s5 raw | s5 snap |
|---|---|---|---|---|
| bwv_846 | 122 | 0.845 | 0.832 | 0.832 |
| bwv_848 | 120 | 0.969 | 0.858 | 0.863 |
| bwv_854 | 120 | 0.944 | **0.900** | **0.903** |
| bwv_856 | 231 | 0.778 | 0.792 | 0.804 |
| bwv_857 | 120 | 0.948 | 0.856 | 0.866 |

Aggregates:
- mean Stage-4 beat-F: 0.897 (unchanged from B13)
- mean Stage-5 raw: 0.769 → **0.847** (+7.8pp)
- mean Stage-5 snap: **0.853** (vs 0.773 without VT)
- Stage 5 pass rate (>= 0.60): 100%
- Stage 5 pass rate at higher bar (>= 0.80): **80%** (4 of 5)

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/dc0khr1u (or geokubj3 retry)

**One piece (bwv_854) clears 0.90 — within striking distance of the spec target.** The variance across pieces is now small (0.80-0.90 range), suggesting the voice tracker is robust.

## Next
- Try `pitch_jump=7` (octave jumps allowed) and longer `time_gap_s` for slow Romantic pieces.
- HMM voice tracker — should pick up the remaining ~5pp.
- Add a learned offset detector — the only way to lift 0.85 → 0.90 reliably across diverse music.
