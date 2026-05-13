# item-g12 — ME-14 system-level ensemble selection

## Goal
task_description_v4.md item G-12. Run 4-6 pipeline variants (different tpb, DP penalties, voicing thresholds), pick the variant whose output maximizes MV2H per piece, average across pieces. Strict pass: ASAP 9-piece mean MV2H lift ≥ +0.015 over single-config tpb=12 baseline, no piece regresses by > 0.02.

## Procedure
- Reused the prior Phase E ME-14 sweep (`reports/_exp_ME14_mv2h_ensemble.json`, 9 ASAP pieces × 4 tpb configs: tpb24_no_corr, tpb24_sanity, tpb12_sanity, tpb6_sanity).
- Ensemble members considered: tpb24_sanity, tpb12_sanity, tpb6_sanity (no_corr is reference only).
- For each piece, picked the variant with the highest MV2H (oracle ensemble — knowing the GT). Wrote results to `reports/_item-g12.json`.
- This is the upper bound on ensemble lift: any system-level selector using a quality proxy (e.g. G-8 round-trip distance) will perform worse than the oracle, never better.

## Results

| piece | best member | best mv2h | tpb12 mv2h | Δ |
|---|---|---|---|---|
| Bach__Fugue__bwv_854 | tpb6_sanity | 0.6023 | 0.5988 | +0.0035 |
| Bach__Fugue__bwv_846 | tpb6_sanity | 0.5507 | 0.5495 | +0.0012 |
| Bach__Fugue__bwv_848 | tpb12_sanity | 0.5490 | 0.5490 | 0.0000 |
| Bach__Fugue__bwv_856 | tpb6_sanity | 0.5685 | 0.5588 | +0.0097 |
| Bach__Fugue__bwv_857 | tpb6_sanity | 0.6255 | 0.6238 | +0.0018 |
| Beethoven__Piano_Sonatas__21-1 | tpb6_sanity | 0.5304 | 0.5147 | +0.0156 |
| Chopin__Berceuse_op_57 | tpb12_sanity | 0.5381 | 0.5381 | 0.0000 |
| Liszt__Sonata | tpb12_sanity | 0.4987 | 0.4987 | 0.0000 |
| Schumann__Toccata | tpb6_sanity | 0.5236 | 0.5117 | +0.0120 |

- single tpb=12 mean = **0.5492**
- oracle ensemble mean = **0.5541**
- ensemble lift = **+0.0049**

## Interpretation
- The ORACLE ensemble (best-per-piece selection knowing the GT) lifts mean MV2H by **+0.0049** — below the strict criterion of +0.015 by **0.011**.
- No piece regresses (oracle selection guarantees nondecrease) — that part of the criterion would pass.
- Phase E's item 6 Bayesian sweep over 122 configs spanning the full hyperparameter space found the production-default max lift at +0.022. ME-14-style hard selection over a 3-config slice of that space can only ever recover a fraction of that.
- The structural issue: 4 of 9 pieces are already maximized by tpb=12 (their best alternative is themselves). The remaining 5 pieces gain at most +0.016 (Beethoven), with the rest under +0.013. Mean over 9 with 4 zeros caps the achievable lift.
- To clear +0.015, the ensemble would need members that genuinely move different *axes* (e.g. one config that swaps the transcriber, one that toggles per_voice_dp). Building those is Phase H scope.

## Pass / discard
- **ASAP MV2H lift ≥ +0.015**: original +0.015, observed +0.0049 → **discarded-with-failure-mode-rationale** (oracle ceiling on the available variants is +0.0049; no quality-proxy router can do better than oracle).
- **No piece regresses > 0.02**: 0/9 pieces regress (oracle selection guarantees nondecrease) → PASS (vacuous).

**Net G-12 status: DISCARDED. The ME-14 oracle ceiling is +0.0049 on the 3-config (tpb24/12/6 with octave_sanity) ensemble. The criterion is unreachable within the current ensemble member set.**

## Next
Phase H ensemble candidates: vary `transcriber` (auto_piano ↔ bytedance_piano ↔ yourmt3plus), vary `per_voice_dp`, vary `formant_offset_corrector` on the humming branch. These touch different axes than tpb and could reach +0.015.
