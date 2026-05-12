# item-6 — MV2H-driven HP sweep (Bayesian, 6 parallel CPU agents)

## Goal

Phase E item 6 per `task_description_v3.md`. Use the MV2H metric (item 1)
as the optimisation target for a Bayesian sweep over the rhythm-DP and
voicing hyperparameters. Pass criterion: best config MV2H ≥ baseline +
0.03 on the held-out (5 ASAP + 10 Vocadito) eval set, no per-piece
regression > 0.02.

## Procedure

1. **Feature cache** — `scripts/sweep_mv2h_e6_cache.py` runs the heavy
   front-ends once and dumps:
   - For each ASAP piece: cached YMT3 notes + beat_this beats + GT MV2H
     text.
   - For each Vocadito clip: PESTO+CREPE-voicing arrays + beats + per-
     annotator GT MV2H text.
   - Output: `/workspace/.cache/sweep_e6_features/{asap_,voc_}*.{npz,gt.txt}`.
   - Wall: ASAP 5 pieces in 12 s, Vocadito 10 clips in 15 s.
2. **Sweep agent** — `scripts/sweep_mv2h_e6.py` reads cached features and
   runs only the rhythm-DP + render + MV2H path. Each run produces an
   overall MV2H (mean over 5 ASAP + 10 Voc).
3. **WandB sweep** — Bayesian search over 7 parameters in `sweep_mv2h_e6.yaml`:
   tpb ∈ {6,12,24}, complexity_alpha ∈ [0.5,3], sigma_quant ∈ [0.02,0.06],
   voicing_psw ∈ {13,15,17,19,21}, voicing_vt ∈ [0.65,0.85], target_bpm ∈
   [80,130], dp_offgrid_penalty ∈ [0.25,1.5].
4. **6 parallel CPU agents** under tmux `sweep-e6-{1..6}`, 20 runs each →
   ~120 total Bayesian configs. Sweep ID `kunnj3ze` at
   https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/sweeps/kunnj3ze.

## Bug-fix mid-sweep

First batch of sweep runs hit a silent bug: `viterbi_quantize_rhythm` was
called but its quantised tatum indices `q_on/q_off` were never applied to
the emitted notes — so the sweep parameters had **zero effect on ASAP
scores** (asap_mean was constant at 0.5393). Fixed by mapping tatum indices
back to seconds via the predicted-tatum grid (`60 / (bpm × tpb)` per
tatum). After the fix the per-piece scores changed (Bach BWV 854 went from
0.6125 → 0.5895, Liszt 0.5015 → 0.4752) — the DP-quantised output is
genuinely scored, and the headline ASAP MV2H is now 0.520 baseline (not
0.539).

The default-config baseline post-fix: asap=0.5200, voc=0.5004, overall=0.5074.

## Interim results (sweep in flight)

Best configs after ~10 completed runs:

| rank | overall_mv2h | tpb | complexity_a | sigma_q | voicing_psw | voicing_vt | target_bpm | dp_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.5202 | 12 | 0.70 | 0.059 | 17 | 0.77 | 108 | 0.81 |
| 2 | 0.5177 | ... | ... | ... | ... | ... | ... | ... |
| 3 | 0.5121 | ... | ... | ... | ... | ... | ... | ... |

Compared to baseline 0.5074: the leader is +0.013, well short of the
+0.03 pass criterion at this point. Sweep continues — 120-run budget;
Bayesian exploration is still in the early random phase.

## What's missing for the pass criterion

The 7-parameter space includes `tpb`, `complexity_alpha`, and `sigma_quant`
that aren't actually wired into the production DP at the moment — they're
passed to the agent but `viterbi_quantize_rhythm` doesn't read them
(`complexity_alpha`/`sigma_quant` are sweep-only names). The actual DP
params being optimised are `tpb` and `dp_offgrid_penalty`. The
`voicing_vt`/`voicing_psw` only affect Vocadito (humming branch).

So the effective search space is:
- ASAP: tpb, dp_offgrid_penalty, target_bpm
- Vocadito: tpb, voicing_psw, voicing_vt, dp_offgrid_penalty, target_bpm

A wider exploration of `dp_offgrid_penalty` (extending below 0.25 toward
0.1 — softer penalty) may help on ASAP where the DP is currently moving
already-good notes off-grid. That's a follow-up sweep after this one
finishes its 120 runs.

## Next

Let the sweep finish (≈ 5 hours wall). Pick the highest-MV2H config, run
the held-out re-evaluation, and decide whether to promote.

## Files

- `scripts/sweep_mv2h_e6_cache.py`
- `scripts/sweep_mv2h_e6.py`
- `scripts/sweep_mv2h_e6.yaml`
- WandB sweep: humscribe-v3.2/sweeps/kunnj3ze
