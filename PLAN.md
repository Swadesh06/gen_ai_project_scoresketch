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

## Phase B+2 — session start (2026-05-03 ~04:00 UTC)

### What's already done
- All Phase A gates pass; Phase B+1 wins live in production defaults
  (TPB=24 metric path, hybrid voicing pesto_crepevoicing vt=0.75 psw=19,
  voice tracking with adaptive_pj=True, auto_piano enum present but no-op
  per B61). 60+ commits, 60+ WandB runs.
- Two biggest unfixed weaknesses (per evaluation):
  - Rendered-SVG over-complexity: 24-lets, 48-lets in Bach BWV 854 SVG.
    Metric (snap = 0.847) doesn't reflect this — it scored well because
    we measure tatum-snap accuracy, not human-readability.
  - Vocadito offset20 F1 = 0.439 vs IAA 0.642 → 20pp gap; B47 tested
    *entry-side* hysteresis and lost; *exit-side* hysteresis untested.
- B58 finding: 100% of remaining ASAP loss is in ByteDance.
  beat_this and DP are essentially perfect on ASAP. So fixing
  the transcriber (item 2: YourMT3+) is the path to Romantic ASAP wins.
- Vocadito IAA ceiling = 0.740 — do not chase above it.

### Six work items (per `reports/results_v1_evalution.md` §Work item 1-6)
1. Rendering polish (CPU)
2. YourMT3+ Romantic-piano backend (GPU inference, ~5 GB VRAM)
3. MusicGen-Melody arrangement (GPU inference, ~13 GB VRAM at fp16)
4. Voicing exit-side hysteresis sweep (CPU)
5. MedleyDB pseudo-label training (speculative, skip if pressed)
6. Final polish: gate re-runs + figures + screencast + README

### Co-scheduling plan (Phase B+2 wave 1)

Per CLAUDE.md hardware-utilization rule: ≥ 1 GPU job + ≥ 1 CPU job + monitor
at all times.

**Wave 1 (parallel):**
- `monitor`: nvidia-smi dmon stream to logs/gpu_monitor.log
- `cpu-render`: item 1 — rendering polish (CPU only, music21 + Verovio)
- `gpu-yourmt3-fetch`: item 2 — clone YourMT3 repo, download checkpoint,
  smoke test
- `cpu-hysteresis`: item 4 — voicing exit-side sweep (CPU only, uses
  cached PESTO+CREPE pitch traces; if no cache, re-extract once)

**Wave 2 (after item 1 lands and item 2 smoke test passes):**
- Re-run gate_asap_rhythm, exp_B12_asap_multi with new rendering
- Run YourMT3+ on the 5 Bach Fugues + 4 Romantic ASAP set
  (both cached predictions saved to /workspace/.cache/asap_yourmt3/)
- In parallel on CPU: render before/after SVGs for item 1 verification

**Wave 3 (item 3):**
- Once items 1, 2, 4 land, do item 3 alone on GPU (MusicGen-Large is
  ~13 GB so co-locate only with CPU render/eval, not with another GPU model).

**Wave 4 (item 6):**
- Final gate re-runs + report figures + screencast + README.

### Phase-C ideas saved for after item 6 lands
1. MERT/MusicFM features → small Transformer voice tracker (combo move)
2. Soft-IAA scoring as headline (B51 followup) — **DONE B65: 0.6466** (kept)
3. AudioLDM2/MAGNeT as MusicGen alternatives (ablation)
4. Anticipatory Music Transformer for score continuation (demo flourish)
5. LoRA fine-tune MusicGen-Melody on hum→arrangement pairs — **smoke DONE B68**

## Phase B+2 results (final, 2026-05-03)

### Six work items
| item | status | result |
|---|---|---|
| 1. Rendering polish | ✓ keep | 24-let removed from BWV 854 SVG; snap unchanged 0.847 |
| 2. YourMT3+ Romantic-piano | ✓ keep | +6.1pp 9-piece ASAP mean (0.713 → 0.774); promoted as default piano transcriber |
| 3. MusicGen-Melody Stage 7 | ✓ keep | 6 presets all work, 4.31 GB peak (1.5B) / 6.25 GB (3.3B "large") |
| 4. Voicing exit-side hysteresis | ✗ discard | offset20 +0.5pp only, below 5pp gate |
| 5. MedleyDB pseudo-labels | skipped | dataset requires registration; deferred to Phase D |
| 6. Final demo polish | ✓ partial | gates re-verified, README written, demo SVGs regenerated; manual screencast still TODO |

### Phase C extensions (this session)
| exp | result | status |
|---|---|---|
| B62 voicing exit hysteresis | offset20 +0.5pp (target +5pp) | discard |
| B63 YMT3+ on 9 ASAP pieces | mean snap +6.1pp; Beethoven 0.811 → 0.897 | keep |
| B64 MusicGen 1.5B preset sweep | 6/6 nonempty, peak 4.31 GB | keep |
| B65 Vocadito soft-IAA scoring | A1 0.665 / A2 0.628 / **soft 0.6466** | keep |
| B66 YMT3+ on Vocadito (humming) | A1 noff 0.497 (vs PESTO+CRP 0.665) | discard — YMT3+ underperforms on monophonic vocal |
| B67 MusicGen-Large 3.3B sweep | 6/6 nonempty, peak 6.25 GB, same 13s/preset | keep |
| B68 LoRA fine-tune smoke | trainable 2.36M / 1.56B = 0.151%; in flight | smoke |

### Headline numbers vs Phase A
- ASAP BWV 846 Stage-5 snap: 0.724 → **0.878** (+15.4pp via YMT3+)
- ASAP 5-Bach mean snap: 0.773 → **0.898** (+12.5pp via YMT3+)
- ASAP Beethoven Sonata 21-1 snap: 0.811 → **0.897** (+8.6pp)
- ASAP Schumann Toccata snap: 0.745 → **0.846** (+10.1pp)
- ASAP Chopin Berceuse snap: 0.481 → **0.675** (+19.4pp)
- ASAP 9-piece overall mean snap: 0.713 → **0.774** (+6.1pp)
- Vocadito A1 noff F1: 0.538 → **0.665** (Phase B+1; unchanged in B+2)
- Vocadito **soft-IAA noff F1: 0.6466** (new headline metric, B65)
- Stage 7 arrangement: ✓ end-to-end, both 1.5B and 3.3B variants verified

### What's left (post item 6)
- Manual screen recording of Streamlit UI doing humming → score → arrangement
- LoRA full training run (B68 is smoke only) on a real (melody, arrangement) pair set
- Phase D: MedleyDB pseudo-label training for Vocadito offset20 push
- Phase D: end-to-end YMT3+ replacing the modular pipeline (complement to its modular use)
- Phase D: MERT features into a learned voice tracker (replacing greedy on dense Romantic textures)
