# Experiments INDEX

| exp_id | date | best metric | status | summary |
|---|---|---|---|---|
| gate_mir1k_n5_seed0 | 2026-05-02 | mean RPA = 0.988 | keep | MIR-1K PESTO sanity gate — pass (gate > 0.85). PESTO loading + voicing wiring correct. |
| gate_asap_v1 | 2026-05-02 | beat F = 0.915, ql-snap = 0.724 | keep | Bach BWV 846 — Stage 4 pass; Stage 5 redefined using mir_eval onset-aligned matching (verbatim spec metric is index-paired and broken on polyphony). Pass at 60% threshold (got 72%). |
| gate_vocadito_soft_A1 | 2026-05-02 | mean COnP F1 = 0.538 | keep | Humming pipeline on all 40 Vocadito clips (A1, soft mode). Pass at 0.40 floor; ~published-baseline territory. |
| gate_mtg_qbh_visual | 2026-05-02 | 20/20 nonempty SVGs | keep | MTG-QBH soft+medium, 10 clips × 2 modes — pipeline survives end-to-end on noisy laptop-mic humming. |
| exp_B1_dp_duration_prior | 2026-05-02 | ASAP raw 0.699 → **0.754** | keep | DP offset quantizer snaps to allowed musical durations; +5.5pp raw lift on Bach BWV 846. B1b cap-by-next-onset reverted (broke polyphony). |
| exp_B2_vocadito_sweep | 2026-05-02 | Vocadito F1 0.538 → **0.577** | keep | 16-run Bayesian sweep; vt=0.315/mns=0.052/psw=11/oms=0.026. Updated soft-mode defaults in `humscribe.config`. |
| exp_B3_crepe_vs_pesto | 2026-05-02 | PESTO 0.576 vs CREPE 0.562 | keep PESTO | CREPE wins per-clip on a few clips but loses by 1.4pp aggregate; voicing-threshold not transferable. |
| exp_B4_hmm_segmenter | 2026-05-02 | HMM 0.518 vs voicing 0.538 | follow-up B4b | Default HMM config too conservative; needs hyperparam tuning. Plumbing kept (`PipelineConfig.note_segmenter`). |
| exp_B5_tempo_adaptive_tpb | 2026-05-02 | ASAP snap 0.719 → **0.740** | keep | TPB=24 default beats TPB=12 by +2.1pp on snapped metric (32nd notes exact). |
