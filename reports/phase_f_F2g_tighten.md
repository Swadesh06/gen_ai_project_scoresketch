# Phase F-2g — tighten F-2e thresholds to remove worst-case voc_8 — NEGATIVE

## Goal

F-2e's production winning config (min_prob=0.30, search_ms=50) ships
**+0.0508 mean off20 F1** on Vocadito but with **voc_8 at Δ −0.053**
exceeding the strict v3 per-piece criterion of > −0.02. F-2g sweeps
`min_prob ∈ {0.30, 0.35, 0.40, 0.45, 0.50}` × `search_ms ∈ {25, 30,
40, 50}` looking for a config where every piece regresses ≤ 0.02 and
mean delta stays positive.

## Procedure

`scripts/eval_f2g_tighten.py`. Caches `track_pitch_hybrid_voicing` +
`segment_pitch_to_notes` outputs for all 40 Vocadito clips once, then
runs the F-2e corrector (`humscribe.pitch.formant_corrector.correct_offsets`)
across the 20 grid cells. Measures mean delta, wins/losses, and
**worst per-piece delta**.

## Grid result — production baseline mean off20 = 0.3433

| min_prob | search_ms | mean Δ | wins/losses | worst clip Δ |
|---|---|---|---|---|
| 0.30 | 25 | +0.0194 | 23/8 | voc_33 (−0.0476) |
| 0.30 | 30 | +0.0312 | 21/8 | voc_33 (−0.0476) |
| 0.30 | 40 | +0.0445 | 26/6 | voc_33 (−0.0476) |
| **0.30** | **50** | **+0.0508** | **28/7** | **voc_8 (−0.0533)** |
| 0.35 | 25 | +0.0190 | 23/8 | voc_33 (−0.0476) |
| 0.35 | 30 | +0.0312 | 21/8 | voc_33 (−0.0476) |
| 0.35 | 40 | +0.0445 | 26/6 | voc_33 (−0.0476) |
| 0.35 | 50 | +0.0502 | 28/7 | voc_8 (−0.0533) |
| 0.40 | 25 | +0.0190 | 23/8 | voc_33 (−0.0476) |
| 0.40 | 30 | +0.0312 | 21/8 | voc_33 (−0.0476) |
| 0.40 | 40 | +0.0452 | 26/5 | voc_33 (−0.0476) |
| 0.40 | 50 | +0.0502 | 28/7 | voc_8 (−0.0533) |
| 0.45 | 25 | +0.0182 | 22/8 | voc_33 (−0.0476) |
| 0.45 | 30 | +0.0304 | 21/8 | voc_33 (−0.0476) |
| 0.45 | 40 | +0.0449 | 26/5 | voc_33 (−0.0476) |
| 0.45 | 50 | +0.0506 | 28/7 | voc_8 (−0.0533) |
| 0.50 | 25 | +0.0182 | 22/8 | voc_33 (−0.0476) |
| 0.50 | 30 | +0.0299 | 21/8 | voc_33 (−0.0476) |
| 0.50 | 40 | +0.0449 | 26/5 | voc_33 (−0.0476) |
| 0.50 | 50 | +0.0506 | 28/7 | voc_8 (−0.0533) |

## Findings

**min_prob is irrelevant in [0.30, 0.50].** Across all 4 search_ms
columns, all 5 min_prob rows produce nearly identical mean delta and
**identical worst-piece delta**. The BiLSTM's confident-but-wrong
peaks all have prob > 0.50 — the threshold isn't gating any of the
false snaps.

**search_ms is the only effective dimension.** Tightening from 50 →
25 ms reduces mean delta from +0.0508 to +0.0194, but improves worst
case only marginally (voc_8 −0.0533 → voc_33 −0.0476). The trade-off
is roughly linear and unfavourable.

**No config meets the strict per-piece criterion.** Best worst case
is voc_33 −0.0476 (search_ms ≤ 40, any min_prob). The −0.02 threshold
is unreachable with this corrector architecture and this BiLSTM model.

## Interpretation

The architecture has a structural failure mode: when the BiLSTM places
a confident peak at the wrong time near the heuristic offset (e.g.
vibrato-tail false-positive, or attack-onset of the *next* note being
detected as the current note's offset), the snap moves the offset
*away* from truth. min_prob can't filter these because they're
above-threshold confident.

What would actually fix this:

1. **Gate on heuristic confidence, not BiLSTM confidence.** Only fire
   the snap when the heuristic's offset is uncertain (low voicing
   exit probability slope, ambiguous transition window). The BiLSTM
   should refine *uncertain* heuristics, not *confident* ones — the
   current corrector gates on the wrong signal.
2. **Train a better BiLSTM.** Vocadito 5-fold gives offset-event F1
   = 0.47 — too low for confident snap-replacement. A larger
   in-domain dataset (e.g. MAESTRO vocals if it existed) might lift
   F1 high enough that confident peaks are right ≥ 95% of the time.
3. **Architecture change.** A regression head outputting an offset
   *correction* (Δt in seconds) rather than a per-frame
   probability, conditioned on the heuristic anchor, would have a
   different failure mode — peaked false-positives wouldn't show up
   as 50 ms shifts because the regression head sees the heuristic
   input.

## Decision

- **Ship F-2e production config (min_prob=0.30, search_ms=50)** with
  default `formant_offset_corrector="off"` per the strict per-piece
  criterion.
- **Stop pursuing the threshold-tightening direction.** F-2g exhausts
  the trivial knob space. The −0.02 worst-case cap is a different
  *kind* of fix.
- Future work (deferred): the "gate on heuristic confidence" path
  above.

## Files

- `scripts/eval_f2g_tighten.py`
- `reports/_phase_f_F2g_tighten.json`
