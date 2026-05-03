# item-4 — voicing exit-side hysteresis (B62 / B62b)

## Goal
B47 tested entry-side hysteresis (which lost). B55 found Vocadito A1 offset20
F1 = 0.439 vs IAA 0.642 (-20pp). Hypothesis: vibrato dips end notes early,
fixed by a lower exit-side threshold so notes persist through dips.

Pass criteria from `task_description_v2.md` §Work item 4:
- Vocadito A1 no_offset F1 ≥ 0.66 (no regression vs 0.665)
- Vocadito A1 offset50 F1 ≥ 0.60 (vs 0.573)
- **Vocadito A1 offset20 F1 ≥ 0.50** (vs 0.439, target +5pp lift)

## Procedure
- `scripts/exp_B62_voc_exit_hysteresis.py` caches PESTO+CREPE traces per clip
  in `/workspace/.cache/vocadito_pitch/<clip>.pkl`, then sweeps `vt_exit` with
  `vt_enter=0.75` fixed. `psw=19, mns=0.052, oms=0.026` from B36b defaults.
- Coarse sweep (B62): `vt_exit ∈ {0.25, 0.35, 0.45, 0.55, 0.65}`.
- Fine sweep (B62b): `vt_exit ∈ {0.65, 0.68, 0.70, 0.72, 0.74}`.
- Each (clip, config) → 6 metrics: A1/A2 × {no_offset, offset50, offset20}.

## Results

### Coarse sweep
| vt_exit | A1 no | A1 o50 | A1 o20 | A2 no | A2 o50 | A2 o20 |
|---|---|---|---|---|---|---|
| 0.25 | 0.637 | 0.512 | 0.400 | 0.593 | 0.475 | 0.363 |
| 0.35 | 0.647 | 0.529 | 0.416 | 0.599 | 0.487 | 0.377 |
| 0.45 | 0.658 | 0.550 | 0.434 | 0.606 | 0.499 | 0.390 |
| 0.55 | 0.662 | 0.561 | 0.440 | 0.613 | 0.512 | 0.398 |
| **0.65** | **0.665** | 0.569 | **0.444** | 0.625 | 0.531 | 0.409 |

### Fine sweep
| vt_exit | A1 no | A1 o50 | A1 o20 | A2 no | A2 o50 | A2 o20 |
|---|---|---|---|---|---|---|
| 0.65 | 0.665 | 0.569 | **0.444** | 0.625 | 0.531 | 0.409 |
| 0.68 | 0.663 | 0.567 | 0.442 | 0.624 | 0.528 | 0.404 |
| 0.70 | 0.665 | 0.568 | 0.440 | 0.627 | 0.532 | 0.402 |
| 0.72 | 0.665 | 0.570 | 0.442 | 0.628 | 0.533 | 0.404 |
| 0.74 | 0.665 | 0.573 | 0.441 | 0.628 | 0.535 | 0.402 |

## Vs criteria

| metric | baseline | best (vt_exit=0.65) | target | met |
|---|---|---|---|---|
| A1 no_offset | 0.665 | **0.665** | ≥ 0.66 | ✓ |
| A1 offset50 | 0.573 | 0.569 | ≥ 0.60 | ✗ (-0.4pp) |
| **A1 offset20** | **0.439** | **0.444** | **≥ 0.50** | **✗ (+0.5pp, far below +5pp target)** |

## Interpretation
The hysteresis is monotone on the curve as expected — lower vt_exit extends
notes, which helps offset-strict matching. But the A1 offset20 lift caps at
+0.5pp. The decision rule (≥ 5pp lift to promote) is not met.

The cause appears structural: PESTO+CREPE voicing on humming has slow,
amplitude-coupled decays, not sharp drops. By the time the smoothed voicing
crosses any sane threshold, the singer has often already started releasing the
note. Lowering `vt_exit` further (e.g. 0.20 or below) would extend the decay
artificially into silence.

## Decision
Discard. Keep `vt_enter == vt_exit == voicing_threshold` (current default).
Real offset-F1 improvement on humming needs a different mechanism — either
a learned offset detector trained on a much larger dataset, or per-note
amplitude-envelope analysis, both of which are Phase-C work.

## Status
discard
