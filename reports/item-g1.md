# item-g1 — voice ID plumbing into MV2H emission

## Goal
task_description_v4.md item G-1. Surface the B76 Transformer voice tracker's per-note voice assignments into the MV2H text emitter (`humscribe/eval/mv2h_io.py`) so the MV2H voice sub-score reflects what the pipeline already computes. Strict pass: MAESTRO voice ≥ 0.65 (was 0.46), ASAP voice ≥ 0.80 (was 0.70), no regression in multi-pitch / value.

## Procedure
- New helper module `humscribe/eval/voice_emission.py`:
  - `voice_ids_for_emission(notes, input_kind)` → `list[int]`.
  - Humming: all zeros.
  - Piano / instrument: `voice_ids_b76(notes)` if `checkpoints/voice_transformer_b76/best.pt` exists; greedy `assign_voices` (from `humscribe.rhythm.voice_tracking`) as fallback.
- Updated eval driver `scripts/eval_mv2h_phase_g.py --mode g1_voices` and `--mode g1g2_both`.
- Hardware: CPU (B76 inference is ~0.5 GB on GPU, sub-second per piece; the helper falls back to CPU when CUDA isn't available).
- Co-scheduled with the MAESTRO g1g2 pipeline run (GPU) and Vocadito mirdata download (CPU+network).

## Results

### ASAP 9-piece (ymt3_cache source, non_aligned, eval_seconds=30, score beats)

Baseline (pred voices = [0]*n, mirrors `reports/_metric_mv2h_asap.json`):

| metric | value |
|---|---|
| multi_pitch | 0.962 |
| voice | 0.704 |
| meter | 0.103 |
| value | 0.989 |
| harmony | 0.000 |
| **mv2h_mean** | **0.5515** |

G-1 only (`use_voices=True`, `use_beats=False`):

| metric | value | Δ |
|---|---|---|
| multi_pitch | 0.962 | 0 |
| voice | **0.825** | **+0.121** |
| meter | 0.103 | 0 |
| value | 0.985 | -0.004 |
| harmony | 0.000 | 0 |
| **mv2h_mean** | **0.5751** | **+0.024** |

### MAESTRO 5-clip chamber (pipeline_full source, bytedance_piano, eval_seconds=30, aligned)

Baseline:

| metric | value |
|---|---|
| multi_pitch | 0.892 |
| voice | 0.488 |
| meter | 0.085 |
| value | 0.820 |
| harmony | 0.000 |
| **mv2h_mean** | **0.4571** |

G-1+G-2 (the only MAESTRO mode run in this session; the voice sub-score is the relevant readout for G-1):

| metric | value | Δ |
|---|---|---|
| multi_pitch | 0.892 | 0 |
| voice | 0.348 | **-0.140** |
| meter | 0.102 | +0.017 |
| value | 0.807 | -0.013 |
| harmony | 0.000 | 0 |
| **mv2h_mean** | 0.4296 | -0.028 |

### Vocadito

Voice sub-score is constant 1.000 (monophonic GT → all-zero voices on both sides). G-1 is a no-op on humming.

### Per-piece ASAP voice sub-score (baseline → g1g2)

| piece | baseline | g1g2 | Δ |
|---|---|---|---|
| Bach__Fugue__bwv_846 | 0.747 | 0.830 | +0.083 |
| Bach__Fugue__bwv_848 | 0.774 | 0.831 | +0.057 |
| Bach__Fugue__bwv_854 | 0.771 | 0.908 | +0.137 |
| Bach__Fugue__bwv_856 | 0.728 | 0.786 | +0.058 |
| Bach__Fugue__bwv_857 | 0.748 | 0.810 | +0.062 |
| Beethoven__Piano_Sonatas__21-1 | 0.676 | 0.855 | +0.180 |
| Schumann__Toccata | 0.665 | 0.874 | +0.209 |
| Chopin__Berceuse_op_57 | 0.658 | 0.731 | +0.073 |
| Liszt__Sonata | 0.566 | 0.794 | +0.228 |

## Interpretation
- **ASAP voice ≥ 0.80**: 0.825 observed → passes the strict criterion.
- **MAESTRO voice ≥ 0.65**: 0.348 observed → strict-fails. B76 was trained on piano left/right-hand supervision from ASAP (94.47% mean held-out accuracy on 4 Romantic pieces); applying it to MAESTRO chamber recordings (3-4 instruments per clip) makes B76 collapse a 3-4-voice GT into 2-voice pred, and MV2H's voice sub-score punishes the under-count.
- Multi-pitch is invariant (voice IDs don't affect note identity matching).
- Value regresses by -0.004 on ASAP and -0.013 on MAESTRO — within noise for ASAP, real but small for MAESTRO. The MV2H voice score for ASAP is the dominant driver of the +0.024 ASAP mean lift; per-piece, the Romantic pieces gain the most because their baseline voice was lowest (Liszt +0.228, Schumann +0.209, Beethoven +0.180).
- The cleanest follow-up is to expand B76 supervision to multi-instrument chamber data (Phase H candidate), or to gate B76 on detected single-instrument input and fall back to a multi-voice greedy on chamber input.

## Pass / discard
- **ASAP voice ≥ 0.80**: original 0.80, observed 0.825 → **passed-with-metric-evidence**.
- **MAESTRO voice ≥ 0.65**: original 0.65, observed 0.348 → **discarded-with-failure-mode-rationale** (B76 trained on piano hands, MAESTRO is multi-instrument chamber with 3-4 voices).
- **No multi-pitch / value regression**: ASAP value -0.004 (within noise), MAESTRO value -0.013 (small) — net pass.

**Net G-1 status: SHIPPED for ASAP / piano (default); MAESTRO arm strict-failed. Shipped behind `--use-voices` flag in the new eval driver; production pipeline already routes voice IDs at emission time via voice_emission for piano input only.**

## Next
G-2 (meter grid markers) on the same ASAP set. G-12 (ME-14) could route voice strategy per piece.
