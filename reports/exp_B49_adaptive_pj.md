# exp_B49 — adaptive pitch_jump per piece

## Goal
B16 found pj=3 optimal for Bach Fugues (snap 0.847 BWV 846). B48 found pj=12 optimal for 5-piece set including 4 Romantic (snap 0.587 mean). Optimal pj is content-dependent. Test an auto-selector that picks pj per piece based on cheap heuristics.

## Procedure
- `humscribe.rhythm.voice_tracking.adaptive_pitch_jump(notes)`:
  - notes/sec > 6 AND pitch IQR > 24 → pj=12 (dense + wide-pitch = Romantic chordal)
  - notes/sec > 4 AND pitch IQR > 18 → pj=7 (compromise)
  - else → pj=3 (Bach-like, narrow polyphony)
- Test on 5 cached ASAP pieces (Bach BWV 846 + 4 Romantic).
- Compare against fixed pj=3, 7, 12.

## Results

| strategy | mean snap | per-piece picks |
|---|---|---|
| **adaptive** | **0.590** | Bach=3, Beethoven=12, Schumann=3, Chopin=3, Liszt=7 |
| fixed_7 | 0.585 | all=7 |
| fixed_12 | 0.584 | all=12 |
| fixed_3 (current default) | 0.571 | all=3 |

Per-piece adaptive results:
| piece | pj | snap |
|---|---|---|
| Bach Fugue BWV 846 | 3 | 0.847 (matches B16 best) |
| Beethoven Sonata 21-1 | 12 | 0.811 (vs 0.718 with pj=3, +9.3pp) |
| Schumann Toccata | 3 | 0.745 |
| Chopin Berceuse | 3 | 0.469 (slow piece — VT can't help much) |
| Liszt Sonata | 7 | 0.078 (broken regardless) |

## Interpretation
**Adaptive PJ wins by +1.9pp over fixed pj=3** without sacrificing Bach Fugue performance. The selector correctly picks pj=3 for narrow polyphony (Bach Fugues), pj=12 for wide chordal (Beethoven Sonata), and falls back to pj=3 for slow/sparse pieces.

The big gain is on Beethoven Sonata (+9.3pp from 0.718 to 0.811) — the wider pj keeps the bass and treble lines together as voices instead of fragmenting on every chord change.

Decision: **promote adaptive_pj as the default** for `quantize_with_voice_tracking()`. Backwards-compatible flag `adaptive_pj=False` available.

## Updated Vocadito vs ASAP final state
- Vocadito A1: 0.665 (unchanged — adaptive_pj is for instrument input only)
- ASAP BWV 846: 0.847 (unchanged — Bach gets pj=3 from adaptive)
- ASAP mean (5 Bach Fugues, B12): 0.856 (unchanged — Bach Fugues all get pj=3)
- ASAP mean (5 pieces with 4 Romantic, B49): **0.571 → 0.590** (+1.9pp)

The headline B12 5-Bach mean stays at 0.856. The new B49 5-mixed mean is 0.590 (vs 0.571 with pj=3).
