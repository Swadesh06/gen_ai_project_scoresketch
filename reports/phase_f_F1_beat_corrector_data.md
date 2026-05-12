# Phase F-1 — beat-corrector training data analysis (full 9 ASAP pieces)

## Goal

Phase F-1 from `reports/PHASE_F_IDEAS.md`. Build a (predicted_beat,
GT_beat) dataset for training a learned beat post-corrector. Targets the
27pp ASAP score-vs-real-beats gap.

## Dataset (full 9 ASAP pieces, eval_seconds=30, tolerance ±0.5 s)

| piece | n_pred | n_gt | matched | mean_shift_s | std_shift_s | max\|shift\|_s |
|---|---|---|---|---|---|---|
| Bach BWV 854 | 103 | 60 | 58 | −0.010 | 0.020 | 0.120 |
| Bach BWV 846 | 133 | 60 | 56 | +0.040 | 0.093 | 0.240 |
| Bach BWV 848 | 206 | 60 | 59 | −0.010 | 0.010 | 0.020 |
| Bach BWV 856 | 69 | 120 | 32 | −0.014 | 0.043 | 0.250 |
| Bach BWV 857 | 253 | 60 | 59 | −0.005 | 0.008 | 0.020 |
| Beethoven 21-1 | 1638 | 76 | 71 | −0.004 | 0.047 | 0.175 |
| **Chopin Berceuse** | **435** | **20** | **20** | **+0.385** | **0.202** | **0.500** |
| Liszt Sonata | 2155 | 60 | 30 | −0.025 | 0.079 | 0.440 |
| Schumann Toccata | 648 | 61 | 60 | +0.008 | 0.068 | 0.244 |

**Total: 445 matched pairs across 9 pieces.**

Aggregate statistics:
- median |shift| = 0.011 s (most beats already correct)
- 90.6% of shifts within ±50 ms
- 92.6% of shifts within ±200 ms

## Interpretation

The 27pp ASAP score-vs-real-beats gap is **NOT** uniformly distributed
across pieces. The failure modes cluster:

1. **Chopin Berceuse** has a systematic +0.385 s mean shift with σ=0.20 s.
   This is the half-tempo octave failure: beat_this detected at ~60 BPM,
   the score is at ~30 BPM. **target_bpm=110 didn't fix it** — the
   distance log2(60/110)=−0.87 is closer than log2(30/110)=−1.87, so
   beat_this picks 60. The +0.385 s shift is approximately half a beat
   at 60 BPM. **Real fix**: detect 2×-multiple between predicted-bpm and
   the score's actual bpm (or note-density heuristic) and halve.

2. **Liszt Sonata** has 30/60 GT beats unmatched at ±0.5 s tolerance —
   50% miss rate, structural. beat_this isn't picking up half the beats.
   Likely the Sonata's wide tempo range exceeds beat_this's training
   distribution. **Real fix**: a learned model with rubato-aware features,
   not a single-shift corrector.

3. **Bach BWV 856** has only 32/120 GT beats matched. The score has 120
   beats in 30 s ≈ 240 BPM; beat_this detected 69 beats ≈ 138 BPM. So
   this is **the opposite** problem: beat_this detected at ½ tempo while
   score is at 2×. Need to handle both directions of the octave issue.

4. **Bach 854, 848, 857; Beethoven 21-1; Schumann Toccata** all have
   median shifts ≤ 0.014 s — beat_this is essentially correct on these.
   A corrector can only hurt these (move correct beats off).

## Implication

A naive "learned beat post-corrector" on this dataset would memorise the
Chopin pattern (n=20 from 1 piece) and learn near-zero shifts elsewhere.
That's not generalisable.

**The actual fix is structural**: detect bpm-octave failures via a
sanity check against the score's beat density or note IOI distribution.
This is closer to a rule-based corrector than a learned one.

**Updated F-1 priority**: implement a tempo-octave sanity-check corrector
(rule-based) before the learned variant. Pass criterion: Chopin Berceuse
ASAP-snap goes from current ~0.66 toward 0.85+.

## Files

- `scripts/prep_beat_corrector_data.py`
- `/workspace/.cache/beat_corrector_data.npz` (445 pairs)
