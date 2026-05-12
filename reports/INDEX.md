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
| exp_B38..B47 batch | many, see batch md | mixed | informative |
| exp_B45b_a2_verify | HMM+hybrid +0.6pp A1 / -0.4pp A2 | discard | discard |
| exp_B48_hdbscan_romantic | -0.4pp on Romantic ASAP | discard | discard |
| **exp_B49_adaptive_pj** | mixed-ASAP 0.571 → **0.590** | **keep** | **keep** |
| exp_B50_bilstm_aug | aug 5x → BiLSTM 0.619 vs voicing 0.648 | discard | discard |
| **exp_B51_vocadito_iaa** | A1<->A2 IAA F1 = **0.740** ceiling | informative | informative |
| exp_B52_hubert_bilstm | HuBERT BiLSTM 0.592 vs voicing 0.648 | discard | discard |
| **exp_B53_oracle_dp** | oracle Liszt = 0.132 (DP-bound!) | informative | informative |
| exp_B54_liszt_dp_sweep | TPB=48 +2.3pp; Liszt unsalvageable | discard | discard |
| **exp_B55_offset_f1** | offset20 F1 = 0.439 vs IAA 0.642 (-20pp) | informative | informative |
| exp_B56_voc_dursnap | tempo-snap durations: flat or worse | discard | discard |
| exp_B57_oms_sweep | current default IS optimum | discard | discard |
| **exp_B58_disambig** | 18.8pp ASAP loss = 100% from ByteDance | informative | informative |
| exp_B59_basicpitch_romantic | -25pp avg vs bd, +9.3pp on Chopin only | informative | informative |
| exp_B60_auto_piano_verify | +5.2pp Chopin / +1.3pp 4-piece mean (kept temporarily) | reverted by B61 | informative |
| **exp_B61_auto_piano_diverse** | bp loses on Debussy (-2.4) and Brahms (-14.5pp) — Chopin was idiosyncratic; auto_piano reverted to no-op | informative | informative |

## Phase B+2 (six demo work items per task_description_v2.md)
| exp_id | metric | result | status |
|---|---|---|---|
| **item-1_rendering** | Bach BWV 854 24-let count: 1 → **0**; 5-Bach mean snap unchanged 0.859 | **keep** | **keep** |
| **exp_B62_voc_exit_hyst** | offset20 +0.5pp (target +5pp); below decision threshold | discard | discard |
| **exp_B63_yourmt3_asap** | 9-piece mean snap 0.713 → **0.774 (+6.1pp)**; Beethoven 0.813 → **0.897** | **keep** | **keep** |
| **item-2_yourmt3plus** | YMT3+ promoted as `auto_piano` default; +12.6pp 3-Romantic mean (ex-Liszt) | **keep** | **keep** |
| **exp_B64_musicgen_presets** | 6/6 presets (1.5B), peak 4.31 GB (4.6× headroom) | **keep** | **keep** |
| **item-3_musicgen** | end-to-end hum→arrange, 6 styles, 1.5B + 3.3B variants both work | **keep** | **keep** |
| **item-4_voicing_exit_hysteresis** | Vocadito offset20 plateau at vt_exit=0.65 with +0.5pp; below 5pp gate | discard | discard |
| **item-6_final_polish** | gates re-validated, README written, demo SVGs regenerated; screencast manual | partial | keep |

## Phase C (Phase B+2 extensions, beyond v2 spec)
| exp_id | metric | result | status |
|---|---|---|---|
| **exp_B65_softiaa_voc** | Vocadito noff_a1 0.665 / a2 0.628 / **soft 0.6466** (new headline metric) | **keep** | **keep** |
| **exp_B66_ymt3_vocadito** | YMT3+ on humming: noff_a1 = 0.497 (-17pp vs PESTO+CRP) | discard | discard |
| **exp_B67_musicgen_large** | 6/6 presets at 3.3B, peak 6.25 GB, 13s/preset same as 1.5B | **keep** | **keep** |
| **exp_B68b_musicgen_lora** | LoRA path attached, loss=NaN (delay pattern unmasked); save+reload OK | partial keep | partial keep |
| **exp_B69_mert_bilstm** | MERT BiLSTM 5-fold CV F1 = 0.517 (overfit; below heuristic 0.665) | discard | discard |
| **exp_B70_mtgqbh_pseudo** (v2) | combined 0.371 / voconly 0.347 / **+2.4pp pseudo gain**, both far below heuristic | discard | discard |

## Phase D (post-spec extensions, ambitious)
| exp_id | metric | result | status |
|---|---|---|---|
| **exp_B70full_mtgqbh_pseudo** | hidden=192 + 40 epochs aggressive config | running | running |
| **exp_B72_aug_bilstm** | BiLSTM + 4× augmentation (pitch shift, time stretch, SpecAug, noise), 80 epochs | running | running |
| **exp_B73_transformer** | Transformer encoder for Vocadito voicing, 5-fold mean F1 = 0.499 (below heuristic 0.665) | discard | discard |
| **exp_B74_musicgen_lora_proper** | LoRA + delay-pattern mask, 200 steps, loss 3.5→2.55, **27% decay**, peak 8.5 GB | **keep** | **keep** |
| **exp_B75_voice_transformer** | Voice tracker on 12 ASAP, **80% mean acc** on Liszt+Schumann | **keep** | **keep** |
| **exp_B76_voice_transformer_scaled** | Voice tracker on 237 ASAP, **94.47% best mean acc** (Liszt 90.8%, Schumann 94.8%, Chopin 94.9%, Beethoven 97.4%) | **keep** | **keep** |
| **exp_B77_musicgen_lora_melody** | LoRA + REAL melody chroma + r=32, 300 steps, loss **3.4→0.73, 69% decay** (2.5× vs B74) | **keep** | **keep** |
| **exp_B78_voice_tracker_integration** | B76 vs greedy on snap-F1: delta=0 (informative null - DP didn't use voice info) | informative | informative |
| **exp_B79_per_voice_dp** | Per-voice DP + B76: **+1.66pp on Chopin Berceuse**, no-op on others | partial keep | keep (Chopin) |
| **exp_B80_lib_check** | Verifies B79 result via library API (per_voice_dp + voice_assigner) | keep | keep |
| **exp_B81_amt_continuation** | AMT (Thickstun stanford-crfm) on monophonic prompt: 0 new events | informative | informative |
| **exp_B82_integration_smoke** | end-to-end pipeline + auto-routing on Chopin Berceuse: **auto_route_fired=True**, MusicXML -90KB | **keep** | **keep** |
| **exp_B83_voice_transformer_aug** | B76 + heavy MIDI augmentation (pitch/time/noise/dropout), 60 epochs | running | running |
| **exp_B84_voice_transformer_big** | Bigger Transformer (d=384, 10 layers, 12M params), 80 epochs | running | running |
| **exp_B85_offset_corrector** | Per-note MLP for Vocadito offset20: -0.66pp vs heuristic | discard | discard |
| **exp_B86_amt_polyphonic** | AMT on polyphonic Bach prompt: 0 new events (rules out OOD; API issue) | informative | informative |
| **exp_B87_pipeline_full_asap** | End-to-end pipeline integration on 9 ASAP pieces | running | running |

## Phase D Production Integration (`humscribe/rhythm/voice_transformer.py`, `pipeline.py:_should_use_per_voice_dp`, `arrange/musicgen.py:lora_adapter`)
- `PipelineConfig(per_voice_dp="auto"|"on"|"off")` — default "auto", routes Chopin-style pieces (nps<10, iqr<24) to B76 + per-voice DP
- `arrange(..., lora_adapter="checkpoints/musicgen_lora_b77/step_300")` — fine-tuned arrangement
- Streamlit Arrange tab auto-discovers saved adapter checkpoints
- Verified end-to-end on Chopin Berceuse (B82): MusicXML -90KB, auto_route fires correctly
- Doc: `reports/PHASE_D_INTEGRATION.md`

## Final headline numbers (after all kept improvements through B+2)
- MIR-1K mean RPA = 0.988 (unchanged)
- ASAP BWV 846 beat-F = 0.915 (unchanged)
- ASAP BWV 846 Stage-5 snap = **0.878** with YMT3+ (Phase A: 0.724, **+15.4pp**); 0.847 with ByteDance
- ASAP mean Stage-5 snap (5 Bach Fugues) = **0.898** with YMT3+ (Phase B+1: 0.859, **+3.9pp**)
- ASAP mean Stage-4 (5 Bach Fugues) = **0.897** (unchanged)
- ASAP Beethoven Sonata 21-1 snap = **0.897** with YMT3+ (Phase B+1: 0.811, **+8.6pp**)
- ASAP Schumann Toccata snap = **0.846** with YMT3+ (Phase B+1: 0.745, **+10.0pp**)
- ASAP Chopin Berceuse snap = **0.675** with YMT3+ (Phase B+1: 0.481, **+19.4pp**)
- ASAP 9-piece overall mean snap = **0.774** with YMT3+ (Phase B+1: 0.713, **+6.1pp**)
- **Vocadito A1 soft F1 = 0.665** (Phase A: 0.538, **+12.7pp** = +24% relative; unchanged through B+2)
- Vocadito A2 soft F1 = **0.630** (Phase A: 0.525, +10.5pp; unchanged)
- Vocadito IAA ceiling: **0.740** — pipeline is 7.5pp below human agreement (no_offset)
- MAESTRO instrument F1 = **0.984** (sanity)
- MTG-QBH 10/10 nonempty = 100%
- **Stage 7 arrangement** (NEW): MusicGen-Melody, 6 style presets, end-to-end hum→arrange working

## Phase C ideas (see PLAN.md for details)
- MERT/MusicFM features for learned segmenter (instead of HuBERT B52)
- Transformer voice tracker for Romantic ASAP (greedy ceiling at Liszt 0.078)
- Soft-IAA scoring as headline (avg of A1/A2 per clip; lower variance)
- AudioLDM2/MAGNeT alternatives to MusicGen
- LoRA-fine-tune MusicGen-Melody on a melody→arrangement pair set
- End-to-end YMT3+ replacing the Stage 2A+4+5 stack on instrument input

## Phase E (this session, 2026-05-12)

| exp_id | metric | result | status |
|---|---|---|---|
| item-1_mv2h_metric | end-to-end MV2H wrapper (Java jar) | shipped + 3 baselines | **keep** |
| item-4_docker_hf | Dockerfile + audiocraft→HF swap | shipped, build deferred | **keep** |
| item-6_mv2h_sweep | Bayesian sweep, 122 runs | top +0.022, tpb=12 promoted | **keep** |
| item-7_ensemble_members | 5/14 members evaluated | mixed (see per-ME) | partial |
| item-8_maestro_demo_regen | rendering polish on chamber clip | integer BPM, key sig, 0×48-lets | **keep** |
| item-3_ddsp_partial | DDSP install attempt | dep chain, deferred | deferred |
| exp_C5_jsb_lora | r=32, 1000 steps | test loss 1.39, plateau | keep (flag) |
| exp_C5b_jsb_lora_r64 | r=64, 1500 steps | step ~250 loss 1.04 (in flight) | running |
| **exp_ME1_pyin** | pYIN diversifier on Vocadito | mean Δ −0.007 | discard |
| **exp_ME4_tonal_prior** | tonal-meter prior on DP | mean Δ −0.006 on ASAP | discard |
| **exp_ME7_anacrusis** | pickup-note detector | 9/9 no-false-positive | keep |
| **exp_ME9_spelling** | line-of-fifths spelling | pitches preserved, accidentals +4.6% | discard |
| **exp_ME10_meter_template** | time-signature ensemble | 1/9 correct | discard |
| **exp_ME14_mv2h_ensemble** | 4-config oracle selection | oracle 0.5494; tpb=12 universal | **keep** |
| **phase_f_F1_octave_sanity** | rule-based beat-octave corrector | **+0.0101 mean MV2H, +0.088 BWV 856** | **keep production** |
| phase_f_F2_formant | formant-band offset detector | fold 1/5 val F1=0.542 (in flight) | running |

## Phase E Production Integration (`humscribe/beat/octave_sanity.py`, `humscribe/eval/`, `humscribe/ensemble/*`)
- `PipelineConfig.tatums_per_beat = 12` (default switched from 24 per ME-14 finding)
- `PipelineConfig.octave_sanity = "auto"` — F-1 rule-based corrector
  (humscribe/pipeline.py applies after beat_this)
- `humscribe/eval/mv2h.py` + `mv2h_io.py` — Java-jar wrapper with text-format converters
- Verified production ASAP 9-piece MV2H = **0.5492** (up from 0.5277 = +0.022)
- Bach BWV 856 single-piece +0.100 from octave-double correction

Doc: `reports/PHASE_E_SESSION_SUMMARY.md`, `reports/PHASE_F_IDEAS.md`
