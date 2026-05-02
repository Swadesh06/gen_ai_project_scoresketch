# gate_mtg_qbh_visual — MTG-QBH visual gate (10 clips × 2 modes)

## Goal
Spec §B.2 reality check on casual/laptop-mic humming. Spec calls this qualitative — visually inspect SVGs for recognizable melodies. We use a structural pass criterion (≥ 80% of clips produce ≥ 1 note) plus eyeball.

## Procedure
- Loader: `humscribe.datasets.mtg_qbh.MTGQBH` (Zenodo-direct; mirdata 1.0.0 doesn't ship `mtg_qbh`).
- Audio: first 10 MTG-QBH clips (q1, q10, q100, q101…q107), 17–30 s each, untrained singers on laptop mics.
- Pipeline: `humscribe.pipeline.transcribe(audio, PipelineConfig(input_kind="humming", mode=<mode>))` for `mode ∈ {soft, medium}`, full Stage 1–6 chain (PESTO → segmenter → beat_this → DP → music21 → SVG).
- Output: `outputs/mtg_qbh_<mode>/<id>.svg` plus a `wandb.Html` log of each SVG for visual inspection in the dashboard.
- Hardware: GPU for PESTO + beat_this, CPU for segmenter + score build. Wall-clock ≈ 6 min for 20 (clip × mode) combinations.

## Results
- **20/20 clips produced ≥ 1 note (100%) — gate PASS** (threshold 80%)
- Soft mode: 39–127 notes per clip, BPM 48–136
- Medium mode: 24–79 notes per clip — fewer, longer notes (the stricter `voicing_threshold=0.50` and `min_note_seconds=0.10` filter out brief glissandi/breath)
- Per-clip SVGs visible in WandB run
- WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/iti7y2w7
- JSON: `reports/_gate_mtg_qbh.json`
- SVGs on disk: `outputs/mtg_qbh_soft/`, `outputs/mtg_qbh_medium/`

## Interpretation
Pipeline survives noisy real-world humming end-to-end. Note-count differences between modes are exactly the intended trade-off in `ModeConfig`: soft accepts more sub-100ms fragments (good for short hummed grace notes, bad for breath); medium consolidates them. BPM detection by beat_this gives reasonable values across the wide range present (q102 sung very slowly at 48 BPM, q103 fast at 136 BPM).

This is a sanity check only — without per-clip note ground truth, we can't compute COnP F1. The Vocadito gate is the quantitative substitute.

## Next
- Pick 5 clips that target known melodies (Twinkle, Frère Jacques etc.); hand-annotate in MuseScore (~10 min/clip per spec §B.2); compute COnP F1 to add a second quantitative measurement on the casual-humming distribution.
- Phase B: this gate naturally tests the same components as Vocadito but on harder audio. Improvements that lift Vocadito F1 should also produce more recognizable MTG-QBH SVGs.
