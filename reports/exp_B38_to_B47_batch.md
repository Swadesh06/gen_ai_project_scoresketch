# exp_B38..B47 — second wave of Phase B+ (post-hybrid-voicing)

After B36b found the +5pp Vocadito win (PESTO+CREPE-voicing hybrid → 0.665), this
batch explored variants and extensions in parallel.

## Summary

| exp | metric | result | vs current best (0.665) | status |
|---|---|---|---|---|
| B38 CREPE-tiny voicing | Vocadito A1 F1 | 0.653 | -1.2pp | ↓ keep tiny option for speed |
| B39 psw beyond 19 | Vocadito A1 F1 | 0.665 (psw=19) | 0 (psw=19 confirmed optimum) | informative |
| B40 MTG-QBH hybrid voicing | note count | similar to PESTO | qualitative | informative |
| B41 Romantic ASAP (partial, killed) | mean snap (4 pieces) | 0.522 (wider_pj7) vs 0.502 (greedy default) | +2pp; Liszt still 0.078 | informative |
| B42b BiLSTM rich (5-fold partial) | fold 1 F1 | 0.582 | -8pp (data-bound) | discard |
| B43 voicing combinations | best F1 | 0.665 (crepe-only) | 0 (matches B36) | informative |
| B44 adaptive vt | best F1 | 0.665 (fixed) | 0 (fixed wins) | informative |
| B45 HMM with hybrid voicing (partial) | best F1 | 0.671 (sigma_v=0.3, p_sus=0.97, p_start=0.05) | **+0.6pp** | tentative keep |
| B46 ASAP no-DP baseline | mean snap | 0.767 vs DP+VT 0.856 | DP+VT adds +9pp | confirms B15 |
| B47 voicing hysteresis (partial) | best F1 | 0.661 (vt_off=0.5) | -0.4pp | discard |

## Findings

1. **Vocadito ceiling is 0.665-0.671** with PESTO+CREPE+segmenter approach. Multiple parallel angles (adaptive vt, voicing combinations, hysteresis, BiLSTM, etc.) confirm this.
2. **B45 HMM+hybrid is the only marginal Vocadito improvement found** (+0.6pp) — but needs multi-piece + A2 verification before promoting to default.
3. **CREPE-tiny is 70% of the speed of CREPE-full at 98% F1** — useful for latency-sensitive setups. Add as `pesto_crepetinyvoicing` option (future work).
4. **DP+voice tracking adds +9pp on ASAP Bach Fugues** (B46 isolated this contribution).
5. **Romantic ASAP is structurally hard** — Liszt Sonata at 0.078 with all VT variants. Needs a fundamentally different approach (learned voice-tracker, or per-piece adaptation with user hints).
6. **BiLSTM remains data-bound** — even with 5-fold CV and rich PESTO+CREPE features, F1 stays 8pp below the heuristic.

## Cumulative Vocadito A1 trajectory

| step | F1 |
|---|---|
| Phase A baseline | 0.538 |
| B2 sweep | 0.577 |
| B22 psw=15 | 0.597 |
| B36 hybrid voicing | 0.650 |
| B36b vt=0.75 psw=19 | 0.665 |
| **B45 HMM+hybrid (partial best)** | **0.671** |

Total from baseline: +13.3pp (+25% relative).

## Decision

- Keep current default: voicing-segmenter + hybrid voicing (vt=0.75, psw=19) producing F1=0.665. Robust across A1 + A2.
- B45's HMM+hybrid F1=0.671 is +0.6pp marginal — pending multi-piece verification, document but not yet promote to default.
- All other variants are flat or worse.

## Next-tier ideas (not yet attempted in this session)

- **MERT/MusicFM features** for the BiLSTM (would unlock B42's data-bound result).
- **Per-piece tempo-aware DP** for slow ASAP pieces (Chopin Berceuse 30 BPM tanks).
- **Real MAESTRO 2018 test split** with audio (not synthesized).
- **Voice clustering with HDBSCAN** for Romantic chordal textures (Liszt).
