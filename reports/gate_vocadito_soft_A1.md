# gate_vocadito_soft_A1 — Vocadito COnP F1 (humming pipeline, soft mode, annotator A1)

## Goal
Phase-2 quantitative gate (the strongest non-toy validation): run the humming pipeline (PESTO → voicing-driven segmenter, no rhythm quantization since the annotations are absolute-time) on every Vocadito clip, compute Correct-Onset-and-Pitch F1 against annotator A1's note transcription. Spec doesn't fix a numeric threshold for v3.2; we set 0.40 as a "not broken" floor — anything significantly below would mean the pipeline isn't usable end-to-end.

## Procedure
- Pipeline: `humscribe.audio_io.load_audio(target_sr=22050)` → `humscribe.pitch.pesto_track.track_pitch_pesto` → `humscribe.pitch.voicing.segment_pitch_to_notes` (median-filter + voicing-thresholded segmenter, see `DESIGN_NOTES.md`).
- Mode: `soft` (`voicing_threshold=0.30`, `min_note_seconds=0.06`, `pitch_smooth_window=7`, `dp_offgrid_penalty=0.5`).
- GT: Vocadito Notes annotator A1 (`Annotations/Notes/vocadito_*_notesA1.csv`). Format: `onset_s, pitch_hz, duration_s`.
- Metric: `mir_eval.transcription.precision_recall_f1_overlap(onset_tolerance=0.05s, pitch_tolerance=50 cents, offset_ratio=None)` — COnP only, ignoring offsets.
- Coverage: all 40 clips.
- Hardware: GPU for PESTO; CPU for everything else.

## Results
- **Mean F1: 0.538** (gate ≥ 0.40 — pass)
- Median F1: ≈ 0.53 (per-clip table in WandB)
- Mean P: ≈ 0.55, Mean R: ≈ 0.55
- Range: F1 between 0.35 (vocadito_33) and 0.69 (vocadito_40)
- WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/ev0z9o6g
- JSON: `reports/_gate_vocadito_soft_A1.json`

## Interpretation
Pipeline works end-to-end on real humming. F1 ≈ 0.54 is consistent with what a simple voicing-thresholded segmenter on top of PESTO can deliver — published Vocadito numbers for similar pipelines fall in 0.55–0.70 (e.g. SwiftF0, MT3 baselines), so we're squarely in the same neighborhood. The voicing segmenter (median filter + 0.5-semitone change detection) is the obvious bottleneck — Phase B priority 2 (proper HMM/Viterbi note segmenter) and priority 3 (learned onset detector) should both lift this.

Per-clip P/R asymmetry tells a story: clips like `vocadito_8` and `vocadito_9` have very high P but low R (we miss notes — likely conservative voicing), while `vocadito_33` and `vocadito_39` have lower P (false-positive notes — likely fragmenting sustained pitches into separate notes when the smoothed contour wobbles). A single `voicing_threshold` doesn't fit both regimes; per-mode hyperparameter sweep (Phase B) should help.

## Next
- WandB sweep: `voicing_threshold ∈ [0.20, 0.65]`, `min_note_seconds ∈ [0.04, 0.16]`, `pitch_smooth_window ∈ {3, 5, 7, 9, 11}`, optimizing for mean Vocadito F1 (annotator A1).
- Compare to annotator A2: re-run with `--annotator A2` to gauge inter-annotator variance.
- Compare modes: re-run with `--mode medium` and `--mode hard`. Expect medium ≈ best on Vocadito (the dataset is mostly clean trained vocals); soft is tuned for casual humming like MTG-QBH.
- Re-run after Phase-B segmenter improvements to track lift.
