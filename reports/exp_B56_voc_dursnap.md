# exp_B56 — Vocadito duration snapping

## Goal
B55 found offset20 F1 = 0.439 vs IAA 0.642 (gap 20pp). Try snapping each note's
duration to the nearest musical subdivision of an estimated tempo. Tempo estimated
from the median inter-onset interval (heuristic: median IOI = 8th-note IOI in humming).

## Procedure
- For each clip, compute median IOI from predicted onsets, set beat = 2 * median_IOI.
- Snap each note duration to nearest of {1/16, 1/12, 1/8, dot-1/8, 1/4, dot-1/4, 1/2, dot-1/2, 1, 1.5, 2} of beat.
- Three strategies: `extend_short` (only short notes), `snap_or_extend` (snap & enforce min 1/16),
  `snap_only` (snap all).
- Compare against `none` (current pipeline).

## Results

| strategy | no_offset | offset20 | offset50 |
|---|---|---|---|
| **none** (default) | **0.665** | **0.439** | **0.573** |
| extend_short | 0.665 | 0.438 | 0.568 |
| snap_or_extend | 0.665 | 0.418 | 0.563 |
| snap_only | 0.665 | 0.417 | 0.562 |

All snapping strategies are flat or slightly worse than no-snap.

## Interpretation
Humming is freely paced — singers don't sing exact 1/8 notes, and the median-IOI tempo
estimate doesn't match the underlying intent of the singer either. Snapping forces durations
toward an artificial grid that the GT also doesn't follow.

The 20pp offset20 gap is real but **duration quantization is the wrong fix**. The fix has
to be in the segmenter itself — better detection of where each note actually ends.

Decision: discard. Move to B57 (segmenter parameter sweep on `onset_merge_seconds` and
`voicing_threshold` — the source of duration noise is likely vibrato dips fragmenting notes).

## Status
discard
