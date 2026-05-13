# Phase G — progress tracker

Generated and maintained by the Phase G agent. Updated per item close.

## Counts

- Stage 1: 0/7
- Stage 2: 0/5
- Stage 3: 0/3
- Stage 4: 0/2
- **Total: 0/17**

## Per-item status

| # | item | stage | status | headline | mv2h_mean Δ | shipped | report |
|---|---|---|---|---|---|---|---|
| G-1 | voice ID plumbing | 1 | pending | | | | |
| G-2 | meter grid markers | 1 | pending | | | | |
| G-3 | F-1b IOI octave detector | 1 | pending | | | | |
| G-4 | same-pitch gap merging | 1 | pending | | | | |
| G-5 | median pitch smoothing | 1 | pending | | | | |
| G-6 | silent-region trimming | 1 | pending | | | | |
| G-7 | pre-recorded demo hums | 1 | pending | | | | |
| G-8 | round-trip self-consistency | 2 | pending | | | | |
| G-9 | confidence-aware output | 2 | pending | | | | |
| G-10 | bar-level diagnostic | 2 | pending | | | | |
| G-11 | render_tpb auto-detect | 2 | pending | | | | |
| G-12 | ME-14 ensemble selection | 2 | pending | | | | |
| G-13 | Lakh MIDI LoRA | 3 | pending | | | | |
| G-14 | multi-take averaging UX | 3 | pending | | | | |
| G-15 | DDSP solo_flute2 retest | 3 | pending | | | | |
| G-16 | C5b subjective artifact | 4 | pending | | | | |
| G-17 | Docker build verification | 4 | pending | | | | |

## Baselines (pre-Phase-G, captured 2026-05-13)

Source: `reports/_metric_mv2h_asap.json` (ymt3_cache, non_aligned, eval_seconds=30):

- ASAP 9-piece MV2H mean: 0.549 (cache path)
- ASAP mean per-axis: multi_pitch 0.962, voice 0.704, meter 0.103, value 0.989, harmony 0.000
- MAESTRO 5-clip MV2H mean: 0.459; per-axis: multi_pitch 0.928, voice 0.463, meter 0.138, value 0.764, harmony 0.000
- Vocadito A1 40-clip MV2H mean: 0.508; per-axis: multi_pitch ~0.80, voice 1.000, meter ~0.01, value ~0.80, harmony 0.000

Production pipeline path (B87b real-beats path) ASAP 9-piece MV2H mean: 0.549.

## Cross-cutting requirements (recap from task_description_v4.md)

- All reports cite all 5 MV2H sub-scores plus mv2h_mean.
- ASAP numbers cite beat-source ("score beats" or "real beats").
- G-1..G-6, G-11 report cites outputs/demos/<piece>_before.svg and <piece>_after.svg.
- OOM-protocol items (G-13 Lakh, MusicGen-Large): dry-run logged at logs/vram_<exp_id>.log.
- Discards close with original threshold + observed value; never "passed with revised threshold".
