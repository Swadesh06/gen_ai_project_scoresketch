# Experiments INDEX

| exp_id | date | best metric | status | summary |
|---|---|---|---|---|
| gate_mir1k_n5_seed0 | 2026-05-02 | mean RPA = 0.988 | keep | MIR-1K PESTO sanity gate. |
| gate_asap_v1 | 2026-05-02 | beat F = 0.915, ql-snap = 0.724 | keep | Bach BWV 846 — Stage 4 + 5 redefined eval; both pass. |
| gate_vocadito_soft_A1 | 2026-05-02 | mean COnP F1 = 0.538 | keep | Phase A baseline; 40 clips A1 soft mode. |
| gate_mtg_qbh_visual | 2026-05-02 | 20/20 nonempty | keep | MTG-QBH soft+medium, 10×2. |
| exp_B1_dp_duration_prior | 2026-05-02 | ASAP raw 0.699 → 0.754 | keep | DP snaps offset to allowed musical durations. +5.5pp. |
| exp_B2_vocadito_sweep | 2026-05-02 | Vocadito F1 0.538 → **0.577** | keep | 16-run Bayesian sweep (+3.9pp). |
| exp_B3_crepe_vs_pesto | 2026-05-02 | PESTO 0.576 vs CREPE 0.562 | keep PESTO | -1.4pp aggregate. |
| exp_B4_hmm_segmenter | 2026-05-02 | HMM 0.518 vs voicing 0.538 | discard | Default HMM worse than voicing. |
| exp_B5_tempo_adaptive_tpb | 2026-05-02 | ASAP snap 0.719 → 0.740 | keep | TPB=24 default (+2.1pp). |
| exp_B6_hmm_sweep | 2026-05-02 | HMM-tuned 0.544 vs voicing 0.577 | discard | HMM ceiling 0.033 below. |
| exp_B7_mtg_qbh_rebaseline | 2026-05-02 | 10/10 | keep | MTG-QBH still passes with B2. |
| exp_B9_vocadito_matrix | 2026-05-02 | A1/soft 0.576, A2/soft 0.525 | keep | 2x3 baseline; soft wins. |
| exp_B10_onset_bilstm | 2026-05-02 | 0.490 (best τ) | discard | 30-clip train too small. |
| exp_B11_voicing_hmm_ensemble | 2026-05-02 | union 0.568 | discard | Errors correlated. |
| exp_B12_asap_multi | 2026-05-02 | mean snap 0.773 across 5 pieces | keep | B1 + B5 wins generalize. |
| exp_B13_tempo_octave | 2026-05-02 | mean Stage-4 0.836 → **0.897** | keep | beat_this(target_bpm=) octave-snap. +6pp mean. |
| exp_B14_maestro_instrument | 2026-05-02 | mean F1 = **0.984** | keep | ByteDance + DP saturated on MAESTRO. |
| exp_B15_voice_tracking | 2026-05-02 | mean Stage-5 snap 0.773 → **0.853** | keep | Greedy voice tracker + per-voice next-onset duration. +8pp generalized. |
