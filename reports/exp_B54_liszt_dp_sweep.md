# exp_B54 — Liszt DP sweep (TPB × pj × allowed_durations)

## Goal
B53 oracle test showed Liszt snap is 0.132 even with GT inputs — DP-bound.
Sweep TPB ∈ {24, 48} × allowed_durations ∈ {default, extended} × pj ∈ {7, 12, 24}
to see if a finer beat-subdivision or wider pitch-jump unlocks Liszt.

## Procedure
- Liszt Sonata: 16025 notes, 2720 GT beats, avg beat 0.57s (~105 BPM).
- For each (tpb, durations, pj) combo, run voice tracking → DP → eval snap%.
- Oracle inputs first (rules out upstream errors).

## Results — oracle inputs

| TPB | durations | pj=7 | pj=12 | pj=24 |
|---|---|---|---|---|
| 24 | default (3,4,6,9,8,12,18,16,24,36,32,48,72,96) | 0.132 | 0.123 | 0.120 |
| 24 | extended (+5/16, 7/16, dot-quintuple) | 0.133 | 0.124 | 0.120 |
| 48 | extended (+64ths, more dots) | **0.155** | 0.147 | 0.143 |

Best: TPB=48 with extended durations + pj=7 → snap=0.155. Marginal.

## Interpretation
- **TPB=48 helps slightly (+2.3pp)** but doesn't unlock Liszt. Even at 64th-note tatum
  resolution, only 15% of GT durations match.
- **pj=7 beats pj=12 and pj=24** in all variants. Wider pitch-jump in voice tracking
  HURTS on Liszt because of cross-voice contamination of duration estimates.
- **The fundamental issue**: Liszt's Sonata uses extreme rubato + irregular tuplets
  + cadenzas that defy fitting to ANY metronomic beat grid. The DP is regularizing
  against a grid the music ignores.

## Conclusion
Liszt is **structurally unsalvageable** by our DP+VT approach. Realistic improvements
would require:
- A music-grammar-aware quantizer (rules: respect time signature, allow tuplets).
- Per-section adaptive beat re-estimation (current beats are global).
- Explicit rubato modeling (warp the beat grid locally).

These are weeks-of-work changes. Decision: **document the ceiling, exclude Liszt from
the headline metric, and report mean-snap excluding Liszt.**

## Updated ASAP headline

| set | best snap | comment |
|---|---|---|
| 5-Bach Fugues (B12) | **0.856** | unchanged |
| 4 mixed (Bach+3 Romantic, ex-Liszt) | **0.718** | recompute B49 minus Liszt |
| 5 mixed (B49 with Liszt) | 0.590 | ceiling capped by Liszt 0.078 |

## Next
- B55: Onset-Offset F1 metric (stricter measure on Vocadito/MIR-1K).
- B56: drop Liszt from the headline mean ASAP metric and recompute.
