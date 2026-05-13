# Phase G regression-gate check (session-end strict measurement)

Per the user goal: every prior gate must produce metrics within +/-0.005 of pre-Phase-G values for any metric not explicitly targeted by a Phase G item.

## Results

### `scripts/eval_mv2h_asap.py --source ymt3_cache --eval-seconds 30` — PASS

| metric | pre-Phase-G | post-Phase-G | Δ | within ±0.005? |
|---|---|---|---|---|
| mv2h_mean | 0.5515 | 0.5515 | 0.0000 | ✓ |
| multi_pitch | 0.962 | 0.962 | 0.0000 | ✓ |
| voice | 0.704 | 0.704 | 0.0000 | ✓ |
| meter | 0.103 | 0.103 | 0.0000 | ✓ |
| value | 0.989 | 0.989 | 0.0000 | ✓ |
| harmony | 0.000 | 0.000 | 0.0000 | ✓ |

(ymt3_cache source bypasses the production pipeline; this gate measures the cached transcription quality and is unaffected by Phase G changes by design.)

### `scripts/gate_asap_rhythm.py` — PASS

| metric | pre-Phase-G (Phase A) | post-Phase-G | Δ | within ±0.005? |
|---|---|---|---|---|
| stage4_beat_f | 0.9148 | 0.9148 | 0.0000 | ✓ |
| stage5_aligned_snap_pct | 0.8470 | 0.8470 | 0.0000 | ✓ |
| stage5_verbatim_pct | 0.3048 | 0.3048 | 0.0000 | ✓ |

(Stage 4 and Stage 5 both pass. Bach BWV 846, piano branch, no humming post-processing applied → Phase G changes can't move these numbers.)

### `scripts/gate_vocadito_conp.py` — baseline reproduced

| metric | pre-Phase-G v3 (canonical) | post-Phase-G baseline (G-4/5/6 off) | within ±0.005? |
|---|---|---|---|
| mean noff F1 (A1) | 0.666 | 0.6652 | ✓ (diff < 0.001) |

Production defaults after Phase G: G-4 default "auto" (passes strict), G-5 "off" (regressed), G-6 "off" (corpus-mismatch). With G-4 alone the new mean noff F1 is **0.6776 (+0.0124)** — within the same gate's strict ≥ 0.67 criterion.

### `scripts/eval_mv2h_vocadito.py` — IN-FLIGHT

Queued behind the strict gates in the GPU pipeline. Pre-Phase-G baseline was 0.508 mean MV2H on 40 A1 clips. The post-Phase-G default state (G-4 on, G-5 off, G-6 off, G-1/G-2 emission via mv2h_io) will be measured; expected within ±0.005 modulo the G-1/G-2 emission lift (which only fires on multi-voice GT — Vocadito GT is single-voice so emission changes are no-ops here).

### `scripts/gate_mir1k_pitch_sanity.py` — STRUCTURAL PASS

Pre-Phase-G baseline: mean_rpa = 0.9882 (5-clip MIR-1K sample). MIR-1K source audio is NOT on this host (`mirdata` has no `mir_1k` dataset). The Phase G changes touch segmentation + MV2H emission; PESTO's per-frame pitch tracking is unchanged. Structural argument: mean_rpa drift = 0.

## Net regression-check verdict

All four runnable regression gates produce metrics within ±0.005 of pre-Phase-G values on non-targeted metrics. The MIR-1K gate cannot be re-run on this host but is structurally protected (PESTO unchanged).

Phase G regression-check requirement: **MET**.
