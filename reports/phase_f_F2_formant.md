# Phase F-2 — formant-band offset detector (in flight)

## Goal

Phase F-2 from `reports/PHASE_F_IDEAS.md`. Train a small BiLSTM on
formant-band (1500-3500 Hz) mel-spectrogram features to detect note
offsets. Targets the 22pp Vocadito offset20 gap (current 0.439 vs IAA
0.642).

## Data + Architecture

- Features cached at `/workspace/.cache/voc_formant/vocadito_*.npz`:
  80-bin log-mel spectrogram, 1500-3500 Hz, 10 ms hop, z-normalised.
- Labels: offset times from Vocadito A1 annotations, with ±1 frame
  smoothing around each offset event.
- Model: 2-layer BiLSTM, 96 hidden units, 0.2 dropout. `humscribe/train/formant_offset.py`.
- Training: `scripts/train_formant_offset.py` runs 5-fold CV across the
  40 clips (32 train / 8 val per fold), 30 epochs per fold, AdamW lr=1e-3,
  positive class weight 40 (offset events are sparse).
- A deeper variant (hidden=128, layers=3) trains in parallel via
  `scripts/train_formant_offset_deep.py`.

## Final results

| fold | val F1 | precision | recall | tps | fps | fns |
|---|---|---|---|---|---|---|
| 1 | 0.542 | 0.506 | 0.583 | 259 | 253 | 185 |
| 2 | 0.486 | 0.414 | 0.590 | 268 | 380 | 186 |
| 3 | 0.415 | 0.347 | 0.516 | 206 | 388 | 193 |
| 4 | 0.453 | 0.386 | 0.548 | 251 | 399 | 207 |
| 5 | 0.430 | 0.374 | 0.506 | 207 | 347 | 202 |

**Base (h=96, l=2): 5-fold mean offset F1 = 0.4652.** High variance
(0.41–0.54) reflects the small-data regime (32 train clips per fold).

**Deep (h=128, l=3): 5-fold mean offset F1 = 0.4697** (+0.005 over base).
Within noise; bigger model doesn't help meaningfully. Confirms the
small-data ceiling.

Both at recall 0.5–0.6, precision 0.35–0.5 (more false positives than
misses) — threshold tuning would help marginally.

These are **offset-event** F1 (no pitch requirement, ±50 ms frame
tolerance) — not directly
comparable to Vocadito's note-level offset20-F1 (0.439) because the
metric definitions differ:

- This F1: detect offset frames within ±5 frame (50ms) tolerance.
- Vocadito offset20-F1: detect offset times within 20% relative-duration
  tolerance, AND requires matching note pitch + onset.

To make the comparison apples-to-apples, the detector must be wired
into the segmenter and `gate_vocadito_conp.py` re-run.

## Production integration plan (Phase F-2 follow-up)

1. Load the best-fold checkpoint into `humscribe.pitch.voicing` (a new
   `segment_pitch_to_notes_with_formant_offset` variant).
2. The function calls PESTO+CREPE-voicing for onsets (existing path)
   and the BiLSTM for offsets (new path).
3. Re-run `scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing
   --mode soft --annotator A1` with and without the new offset path.
4. Decision rule (item 7 pass criterion): if offset20-F1 improves by ≥
   5pp without regressing no-offset F1 by ≥ 1pp, ship as default.

## Caveats

- Vocadito is tiny (40 clips) — 5-fold CV minimum is 32 training clips.
  The detector may overfit on this scale. Phase F-2b should pretrain on
  MIR-ST500 (partial download 27/30 in progress) for a 12× data lift.
- The current 80-bin mel resolution may be too narrow at 1500-3500 Hz.
  Vocal formant peaks are typically 200-400 Hz wide; at 80 bins between
  1500-3500 Hz that's 25 Hz per bin — adequate but not over-resolved.

## Files

- `humscribe/train/formant_offset.py`
- `scripts/prep_formant_features.py`
- `scripts/train_formant_offset.py` (5-fold CV)
- `scripts/train_formant_offset_deep.py` (hidden=128, layers=3)
- `reports/_phase_f_F2_formant.json` (will land when CV completes)
- `/workspace/.cache/voc_formant/` (40-clip feature cache)
