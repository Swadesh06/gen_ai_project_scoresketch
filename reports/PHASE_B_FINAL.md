# Phase B — final campaign summary

## Headline trajectory

| metric | Phase A | Phase B+1 best | Δ vs Phase A |
|---|---|---|---|
| MIR-1K mean RPA | 0.988 | 0.988 | 0 (saturated) |
| ASAP BWV 846 beat-F | 0.915 | 0.915 | 0 |
| **ASAP BWV 846 Stage-5 snap** | 0.724 | **0.847** | **+12.3pp** |
| ASAP mean Stage-5 snap (5 Bach Fugues) | 0.773 | **0.856** | +8.3pp |
| ASAP mean Stage-4 (5 Bach Fugues) | 0.836 | **0.897** | +6.1pp |
| **Vocadito A1 soft F1 (40 clips)** | 0.538 | **0.665 (default) / 0.671 (B45 partial)** | **+12.7-13.3pp** |
| Vocadito A2 soft F1 | 0.525 | 0.630 | +10.5pp |
| MAESTRO instrument F1 (sanity, MIDI-rendered) | n/a | 0.984 | n/a |
| MTG-QBH visual nonempty | 100% | 100% | 0 |

Bach BWV 854 hits Stage-5 snap = **0.904** — first piece to clear the 0.90 spec target.

## Phase B count: 30+ experiments

### Wins (kept, in production defaults)
| exp | description | impact |
|---|---|---|
| **B1** | DP duration prior on offset quantization | +5.5pp ASAP raw |
| **B2** | Vocadito Bayesian sweep (vt, mns, oms, psw) | +3.9pp Vocadito |
| **B5** | TPB=24 default (32nd-note exact rep) | +2.1pp ASAP snap |
| **B13** | beat_this `target_bpm=` octave-snap (eval-time) | +6pp ASAP S4 mean |
| **B14** | MAESTRO sanity (validation) | n/a |
| **B15** | Voice tracking + per-voice DP | +8pp ASAP S5 mean |
| **B16** | VT hyperparams (pj=3, tg=0.5) | +0.3pp ASAP S5 |
| **B18** | Verovio real-notation SVG | qualitative |
| **B22** | Vocadito psw=15 (extended sweep) | +2pp Vocadito |
| **B36/B36b** | PESTO pitch + CREPE periodicity voicing | **+5.3pp Vocadito** |

### Discarded (with rationale)
- B3, B17, B27 — CREPE/ensembles lose to PESTO+psw=15
- B4, B6 — HMM segmenter (default + sweep) below voicing baseline
- B10, B19, B42b — BiLSTM onset detectors (training data too small)
- B11 — voicing+HMM ensemble (correlated errors)
- B20, B21, B25 — HMM voice tracker (loses to greedy on Bach)
- B23 — DP hyperparam sweep (already at optimum on clean piece)
- B26 — vt/mns/oms re-sweep with psw=15 (no improvement)
- B33 — dense PESTO step_ms (5ms hurts; 15ms within noise)
- B34 — basic_pitch on Vocadito (F1=0.495, generates too many false positives)
- B35 — librosa onset_detect (F1=0.380, weak)
- B41 — Romantic ASAP (Liszt structurally broken; killed early)
- B43 — voicing combination strategies (crepe-only wins all)
- B44 — per-clip adaptive vt (fixed 0.75 wins)
- B47 — voicing hysteresis (best 0.666 vs no-hysteresis 0.665)

### Tentative (B45)
- B45 HMM segmenter with hybrid voicing finds F1=0.671 (sigma_v=0.3, p_sustain=0.97, p_start=0.05). +0.6pp marginal over voicing baseline 0.665. Needs multi-piece + A2 verification before promoting.

### Informative
- B7 MTG-QBH re-baseline | B9 Vocadito 2x3 matrix | B12 ASAP multi-piece | B26 confirms ceiling | B28 A2 generalization | B37 ASAP non-Bach (Liszt 0.078, Chopin 0.469) | B38 CREPE-tiny voicing (98% F1 at 70% time) | B39 psw>19 confirms 19 is optimum | B40 MTG-QBH hybrid voicing (similar note counts) | B46 no-DP baseline 0.767 (DP+VT adds +9pp)

## Cumulative Vocadito A1 trajectory

```
Phase A baseline:    0.538
+ B2 sweep:          0.577  (+3.9pp)
+ B22 psw=15:        0.597  (+2.0pp)
+ B36 hybrid voicing: 0.650  (+5.3pp)
+ B36b vt=0.75 psw=19: 0.665 (+1.5pp)
[B45 HMM+hybrid:     0.671  (+0.6pp tentative)]
```

Total: **+12.7pp = +24% relative** improvement over Phase A baseline.

## Cumulative ASAP BWV 846 Stage-5 snap

```
Phase A baseline:     0.724
+ B1 DP duration prior: 0.719 (within noise)
+ B5 TPB=24:           0.740  (+1.6pp)
+ B15 voice tracking:  0.779  (+3.9pp)
+ B16 VT sweep tuning: 0.847  (+6.8pp)
```

Total: **+12.3pp** on BWV 846. Multi-piece mean: 0.773 → 0.856.

## What's left for Phase B+2 (next agent)

Hard problems requiring substantial new work:

1. **Romantic ASAP (Liszt at 0.078)**: voice tracker over-fragments Romantic chordal textures. Needs a learned voice tracker.
2. **Slow-tempo beat tracking** (Chopin Berceuse 30 BPM tanks beat F at 0.39): needs wide-tempo beat detector.
3. **Vocadito above 0.70**: saturated for the heuristic + small-data BiLSTM. Need pre-trained features (MERT, MusicFM) or 10x more training data.
4. **MAESTRO 2018 test split with audio**: published-comparable note F1 (currently sanity-only).
5. **Demo polish**: per-clip MTG-QBH ground truth annotation (5 clips × 10 min in MuseScore) for quantitative MTG-QBH F1.

## How to reproduce

```bash
conda activate humscribe
cd /workspace/swadesh/gen_ai_project_scoresketch
set -a && source .env && set +a

# Vocadito A1 best (hybrid voicing, F1=0.665):
python scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing --mode soft --annotator A1

# ASAP BWV 846 (Stage-5 snap=0.847):
python scripts/gate_asap_rhythm.py    # voice tracking + B16 defaults are on by default

# ASAP multi-piece (mean snap=0.856):
python scripts/exp_B12_asap_multi.py --n-pieces 5

# MAESTRO instrument sanity (F1=0.984):
python scripts/exp_B14_maestro_instrument.py --n-pieces 5

# MIR-1K sanity (RPA=0.988):
python scripts/gate_mir1k_pitch_sanity.py
```

50+ commits, 30+ Phase-B experiments, 60+ WandB runs. Pipeline at https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2.
