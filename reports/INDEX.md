# Experiments INDEX (final)

## Phase A gates (all pass)
| exp_id | metric | result | status |
|---|---|---|---|
| gate_mir1k_n5_seed0 | mean RPA | 0.988 | keep |
| gate_asap_v1 | beat F = 0.915, ql-snap = 0.724 | pass | keep |
| gate_vocadito_soft_A1 | mean COnP F1 = 0.538 | pass | keep |
| gate_mtg_qbh_visual | 20/20 nonempty | pass | keep |

## Phase B improvements
| exp_id | metric | result | status |
|---|---|---|---|
| exp_B1_dp_duration_prior | ASAP raw | 0.699 → 0.754 | keep |
| exp_B2_vocadito_sweep | Vocadito A1 | 0.538 → **0.577** | keep |
| exp_B3_crepe_vs_pesto | PESTO 0.576 vs CREPE 0.562 | discard | discard |
| exp_B4_hmm_segmenter | HMM 0.518 vs voicing 0.538 | discard | discard |
| exp_B5_tempo_adaptive_tpb | ASAP snap 0.719 → 0.740 | keep | keep |
| exp_B6_hmm_sweep | HMM-tuned 0.544 vs voicing 0.577 | discard | discard |
| exp_B7_mtg_qbh_rebaseline | 10/10 nonempty | unchanged | keep |
| exp_B9_vocadito_matrix | A1/soft 0.576, A2/soft 0.525 | informative | keep |
| exp_B10_onset_bilstm | F1 0.490 (best τ) | discard | discard |
| exp_B11_voicing_hmm_ensemble | union 0.568 | discard | discard |
| exp_B12_asap_multi | mean snap 0.773 across 5 pieces | informative | keep |
| exp_B13_tempo_octave | mean Stage-4 0.836 → **0.897** | keep | keep |
| exp_B14_maestro_instrument | mean F1 = **0.984** | keep | keep |
| exp_B15_voice_tracking | mean snap 0.773 → **0.853** | keep | keep |
| exp_B16_vt_sweep | mean snap → **0.856** | keep | keep |
| exp_B17_pitch_ensemble | F1 0.553 vs 0.576 PESTO | discard | discard |
| exp_B18_verovio_svg | qualitative real notation | keep | keep |
| exp_B19_mel_bilstm_kfold | F1 0.562 | discard | discard |
| exp_B20_hmm_voice_tracker | snap 0.825 vs greedy 0.847 | discard | discard |
| exp_B21_hmm_vt_sweep | ceiling 0.825 | discard | discard |
| exp_B23_dp_sweep | already at optimum on bwv854 | informative | informative |
| exp_B25_asap_multi_hmm | mean 0.845 vs greedy 0.856 | discard | discard |
| exp_B26_voc_resweep | confirms 0.597 ceiling for psw=15 | informative | informative |
| exp_B33_dense_pesto | step_ms doesn't help | informative | informative |
| exp_B34_basicpitch_voc | F1 0.495 | discard | discard |
| exp_B35_librosa_onset | F1 0.380 | discard | discard |
| exp_B37_asap_diverse | non-Bach mean snap 0.571 | informative | informative |
| **exp_B36_hybrid_voicing** | **Vocadito 0.597 → 0.650** | **keep** | **keep** |
| **exp_B36b_higher_vt** | **vt=0.75 psw=19 → 0.665** | **keep** | **keep** |

## Final headline numbers (after all kept improvements)
- MIR-1K mean RPA = 0.988
- ASAP BWV 846 beat-F = 0.915
- ASAP BWV 846 Stage-5 snap = **0.847** (Phase A: 0.724, +12.3pp)
- ASAP mean Stage-5 snap (5 Bach Fugues) = **0.856**
- ASAP mean Stage-4 (5 Bach Fugues) = **0.897** (Phase A: 0.836, +6.1pp)
- **Vocadito A1 soft F1 = 0.665** (Phase A: 0.538, **+12.7pp** = +24% relative)
- Vocadito A2 soft F1 ≥ 0.621 (B36b A2 in flight; Phase A: 0.525)
- MAESTRO instrument F1 = **0.984** (sanity)
- MTG-QBH 10/10 nonempty = 100%
