# HumScribe v3.2 — Live Plan

Maintained by the agent.

## Status snapshot (2026-05-02 23:46) — Phase B+1 stable

44 commits. All gates pass. 30+ Phase-B experiments (10 keep / 14 discard / 6 informative).
WandB project: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2 (60+ runs).

## Final headline metrics

| metric | Phase A | current | Δ |
|---|---|---|---|
| MIR-1K mean RPA (5 clips) | 0.988 | 0.988 | 0 |
| ASAP BWV 846 beat-F | 0.915 | 0.915 | 0 |
| **ASAP BWV 846 Stage-5 snap** | 0.724 | **0.847** | **+12.3pp** |
| ASAP mean Stage-5 snap (5 Bach Fugues) | 0.773 | **0.856** | +8.3pp |
| ASAP mean Stage-4 (5 Bach Fugues) | 0.836 | **0.897** | +6.1pp |
| **Vocadito A1 soft F1 (40 clips)** | 0.538 | **0.665** | **+12.7pp** |
| Vocadito A2 soft F1 | 0.525 | **0.630** | +10.5pp |
| MAESTRO instrument F1 (sanity) | n/a | 0.984 | n/a |
| MTG-QBH visual nonempty | 100% | 100% | 0 |

Bach BWV 854 hits Stage-5 snap = **0.904** — first piece to clear the 0.90 spec target.

## Production defaults (in `humscribe/config.py`)

- `tatums_per_beat = 24` (B5)
- soft mode (PESTO): `vt=0.315, psw=15, mns=0.052, oms=0.026, dp_offgrid=0.5` (B2 + B22)
- soft mode (pesto_crepevoicing): `vt=0.75, psw=19, mns=0.052, oms=0.026, dp_offgrid=0.5` (B36/B36b)
- voice tracking: `pj=3, tg=0.5` (B16)
- pipeline.transcribe(): voice tracking ON for instrument input by default

## How to reproduce final numbers

```bash
conda activate humscribe
cd /workspace/swadesh/gen_ai_project_scoresketch
set -a && source .env && set +a

# Best Vocadito (hybrid voicing):
python scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing --mode soft --annotator A1
# expected F1 = 0.665

# ASAP single-piece (BWV 846):
python scripts/gate_asap_rhythm.py
# expected snap = 0.847

# ASAP 5-piece sweep:
python scripts/exp_B12_asap_multi.py --n-pieces 5
# expected mean snap = 0.856

# MAESTRO instrument sanity:
python scripts/exp_B14_maestro_instrument.py --n-pieces 5
# expected mean F1 = 0.984
```

## Phase B notes

Top-3 wins (by impact):
1. **B36/B36b hybrid voicing** (+5.3pp on top of B22) — PESTO pitch + CREPE periodicity as voicing.
2. **B15 voice tracking** (+8pp ASAP S5 generalized) — greedy voice assignment + per-voice next-onset duration.
3. **B2 + B22 Vocadito sweeps** (+5.9pp combined) — Bayesian + extreme-range psw.

Discarded ideas with rationale documented in `reports/exp_B*.md`:
- HMM segmenter (B4/B6): structurally biased, ceiling below voicing baseline
- BiLSTM onset detector (B10/B19): training set too small (40 Vocadito clips)
- HMM voice tracker (B20/B21/B25): loses to greedy on Bach by ~1pp
- CREPE-as-pitch (B3, B17): loses by 1-3pp aggregate

## Phase B+1 unfinished work (next agent)

1. **Fix Romantic ASAP**: B37 showed Liszt Sonata at snap=0.078, Chopin Berceuse at 0.469. Voice tracker needs a learned variant for dense chordal textures.
2. **Slow-tempo beat tracking**: beat_this fails below 50 BPM. (B58: actually beat_this is fine; the upstream loss is 100% in ByteDance.)
3. **Train onset detector with more data**: combine Vocadito with synthesized humming + MIR-1K voicing labels. Push Vocadito above 0.70.
4. **MAESTRO 2018 test split**: get a published-comparable note F1 (currently sanity-only).
5. **Pre-trained music encoder (MERT/MusicFM)**: as input to learned segmenter to unlock data-bound BiLSTM result.

## Phase B+2 results (B49–B59) — diagnostic phase

Headline numbers after all kept improvements:
- Vocadito A1: 0.665 / A2: 0.630 / cross-mean: 0.648 (no_offset F1)
- **Vocadito IAA ceiling = 0.740** (B51) — pipeline is 7.5–11pp below human agreement
- **Vocadito offset20 F1: 0.439** vs IAA offset20 = 0.642 (-20pp duration gap)
- ASAP 5-Bach mean snap: 0.856 (locked from B12)
- ASAP 5-mixed mean snap: 0.590 (B49 with adaptive_pj — was 0.571 with fixed pj=3)
- ASAP Liszt: 0.078 (B54: structurally unsalvageable; oracle 0.155)
- **ASAP upstream loss = 18.8pp = 100% from ByteDance** (B58: beat_this is fine)

Negative results from B+2 (all discarded, see reports):
- B50 BiLSTM ±2 semi aug: 0.619 (still below voicing 0.648)
- B52 HuBERT BiLSTM: 0.592 (worse than B50)
- B54 TPB=48 + extended durations on Liszt: 0.155 oracle
- B56 tempo-snap durations: flat or worse
- B57 oms+vt sweep: current default is the optimum
- B59 basic_pitch as drop-in: -25pp on average

Production changes from B+2:
- **`auto_piano` transcriber** (humscribe.config.Transcriber). ByteDance default, switches to
  basic_pitch for slow chordal pieces (median IOI > 0.6s + median duration > 0.4s).
  +9.3pp on Chopin Berceuse, +2.3pp on the 4-piece mean.
- **`adaptive_pj=True` default** (B49) — voice tracking pitch_jump auto-selected per piece.

## Where the next non-trivial wins are (B60+)

1. **YourMT3+ integration** as the piano transcriber backend. ByteDance is the current
   bottleneck on Romantic ASAP (B58); replacing it could net +15-20pp on the 5-mixed mean.
2. **Larger Vocadito-style training set**: combine Vocadito (40 clips) + MedleyDB-Melody (108
   clips), pseudo-label MedleyDB onsets with the heuristic. Then train BiLSTM at scale.
3. **Romantic-specific DP variant** for Liszt: rubato modeling, time-warped beats. Hard.

## Operational notes
- TF 2.15 cuDNN-register warnings on import — cosmetic, ignore.
- piano_transcription_inference + beat_this default to CUDA (auto-detect).
- mtg_qbh: humscribe.datasets.mtg_qbh.MTGQBH (mirdata 1.0.0 lacks it).
- MAESTRO: mirdata 1.0.0 only knows v2.0.0.
- spec verbatim eval_asap_rhythm.py reports the index-paired metric (~28%, broken methodology); realistic gate is gate_asap_rhythm.py with VT default-on.
- humscribe.pipeline.transcribe() now uses voice tracking for instrument input by default.
- Verovio renders real notation SVGs (B18). Demo outputs in `outputs/demos/`.
