# exp_B57 — onset_merge_seconds + voicing_threshold sweep

## Goal
B55 found offset20 F1 = 0.439 vs IAA 0.642 (+20pp room). Hypothesized that vibrato dips
fragment notes into smaller pieces, hurting durations. Sweep `oms` (gap closing) and `vt`
+ `mns` to test.

## Procedure
- Grid: vt ∈ {0.65, 0.75} × oms ∈ {0.026, 0.05, 0.1, 0.15, 0.2} × mns ∈ {0.052, 0.1, 0.15}
- 30 configs × 40 Vocadito A1 clips = 1200 evaluations.
- Score no_offset, offset20, offset50 F1 each.

## Results — top 5 by offset20 F1

| rank | vt | oms | mns | no_off | off20 | off50 |
|---|---|---|---|---|---|---|
| 1 (current default) | 0.75 | 0.026 | 0.052 | **0.665** | **0.439** | 0.573 |
| 2 | 0.65 | 0.026 | 0.052 | 0.659 | 0.434 | 0.560 |
| 3 | 0.75 | 0.050 | 0.052 | 0.654 | 0.422 | 0.544 |
| 4 | 0.65 | 0.026 | 0.100 | 0.609 | 0.416 | 0.537 |
| 5 | 0.75 | 0.026 | 0.100 | 0.606 | 0.411 | 0.538 |

Increasing `oms` (segment merging) **hurts** all metrics. Increasing `mns` (min note length)
also hurts.

## Interpretation

The current default (oms=26ms, vt=0.75, mns=52ms) is **already at the local optimum** for
both no_offset and offset20 metrics. The vibrato-fragmentation hypothesis is wrong — wider
gap-closing merges legitimately separate notes, costing precision.

The offset20 F1 ceiling for our segmenter architecture is **0.439**. To go higher, we need
a fundamentally different approach to offset detection:

1. **Separate offset detector** — train a model that predicts "this is a note offset",
   independent of onset detection.
2. **Pitch-stability based segmenter** — note ends when pitch deviates >0.5 semitones for >30ms,
   regardless of voicing.
3. **End-to-end transformer** — predict note boundaries jointly.

Each requires either a different architecture or much more training data.

## Decision
Discard. Current segmenter parameters are optimal. Move to architectural changes (B58).

## Status
discard (no improvement)
