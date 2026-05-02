# gate_mir1k_n5_seed0 — MIR-1K PESTO sanity gate

## Goal
Confirm the Stage 2-B.1 sanity gate from spec §B.3: PESTO running on its own training distribution (MIR-1K) should produce mean RPA > 0.85 across a small sample. If this fails, the bug is in our loading/voicing wiring, not in PESTO. Pass = our I/O + PESTO wrapper is correct before any other Stage 2 / Stage 3 work.

## Procedure
- Script: `scripts/gate_mir1k_pitch_sanity.py` (WandB-instrumented wrapper around the verbatim `scripts/eval_mir1k_pitch_sanity.py` logic).
- Dataset: 5 random clips from `~/datasets/mir1k/MIR-1K/Wavfile/` (seed=0, deterministic).
- For each clip: extract right channel (vocal), run `humscribe.pitch.pesto_track.track_pitch_pesto`, interpolate predictions to the GT 20 ms grid, compute `mir_eval.melody.raw_pitch_accuracy` with `cent_tolerance=50`, voicing taken from `gt_midi > 0`.
- Hardware: CPU-side pre/post + Blackwell GPU for PESTO (torch 2.11.0+cu128, sm_120). VRAM peak: well under 1 GB.

## Results
| clip | RPA |
|---|---|
| stool_4_01.wav | 0.989 |
| bobon_5_10.wav | 0.997 |
| leon_4_01.wav | 0.991 |
| titon_3_05.wav | 0.991 |
| bug_5_10.wav | 0.973 |

- **Mean RPA: 0.988** (gate: > 0.85 — pass)
- median 0.991, p25 0.989, p75 0.991
- WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/9s3wfty8
- JSON: `reports/_gate_mir1k.json`

## Interpretation
PESTO + our wrapper deliver near-published numbers (paper reports ~0.90 RPA on MIR-1K; we see 0.99 on a 5-clip sample, which is well within the per-clip variance of MIR-1K and slightly above the published average). The right-channel extraction, mono load, sample-rate handoff, and millisecond-to-second time conversion are all correct. Any future failure on Vocadito or noisy humming is downstream of pitch tracking.

## Next
ASAP rhythm gate (Stage 4 + 5). Will need ByteDance piano weight download (~330 MB on first call) plus beat_this checkpoint (~70 MB). After that, Vocadito quantitative + MTG-QBH visual.
