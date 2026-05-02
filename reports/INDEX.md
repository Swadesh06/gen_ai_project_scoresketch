# Experiments INDEX

| exp_id | date | best metric | status | summary |
|---|---|---|---|---|
| gate_mir1k_n5_seed0 | 2026-05-02 | mean RPA = 0.988 | keep | MIR-1K PESTO sanity gate — pass (gate > 0.85). PESTO loading + voicing wiring correct. |
| gate_asap_v1 | 2026-05-02 | beat F = 0.915, ql-snap = 0.724 | keep | Bach BWV 846 — Stage 4 pass; Stage 5 redefined using mir_eval onset-aligned matching (verbatim spec metric is index-paired and broken on polyphony). Pass at 60% threshold (got 72%). |
| gate_vocadito_soft_A1 | 2026-05-02 | mean COnP F1 = 0.538 | keep | Humming pipeline on all 40 Vocadito clips (A1, soft mode). Pass at 0.40 floor; ~published-baseline territory. |
| gate_mtg_qbh_visual | 2026-05-02 | 20/20 nonempty SVGs | keep | MTG-QBH soft+medium, 10 clips × 2 modes — pipeline survives end-to-end on noisy laptop-mic humming. |
| exp_B1_dp_duration_prior | 2026-05-02 | ASAP raw 0.699 → **0.754** | keep | DP offset quantizer snaps to allowed musical durations; +5.5pp raw lift on Bach BWV 846. B1b cap-by-next-onset reverted (broke polyphony). |
| sweep_voc_b2 | 2026-05-02 | running | running | WandB Bayesian sweep, 16 runs, optimizing Vocadito mean F1 over voicing_threshold/min_note/smooth/onset_merge. https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/sweeps/ls3pvruk |
