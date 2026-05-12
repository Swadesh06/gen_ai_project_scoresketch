# Phase F-2b — MIR-ST500 pretrain prep (data ready, training deferred)

## Goal

Phase F-2 (formant offset detector on Vocadito) plateaued at 5-fold mean
F1 ≈ 0.47 with high variance (0.41–0.54) due to small data — only 32
training clips per fold. F-2b pretrains the same architecture on MIR-ST500
(500 pop songs, ~30 hours total) for a 12× data lift, then fine-tunes
back to Vocadito.

## What's ready

- **84 / 100 MIR-ST500 mp3s downloaded** via yt-dlp (`scripts/prep_mirst500_partial.py`).
  16 unavailable due to YouTube takedowns. Files at
  `/workspace/datasets/mirst500/audio_partial/<id>.mp3`.
- **Formant features extracted for all 84 songs** via
  `scripts/prep_mirst500_formant.py`. 80 MB total cache at
  `/workspace/.cache/mirst500_formant/<id>.npz` (80-bin mel,
  1500-3500 Hz, 10 ms hop, normalised).
- **Labels available** in
  `/workspace/datasets/mirst500/repo/MIR-ST500_20210206/MIR-ST500_corrected.json`:
  per-song list of `[onset_s, offset_s, midi_pitch]` triples,
  ~200-400 notes per song.

## What remains (next session)

1. Write `scripts/train_formant_offset_mirst500.py` that:
   - Loads formant features + labels for each of the 84 songs.
   - Window each song into 10 s clips (matches the F-2 Vocadito window).
   - Train the F-2 `FormantOffsetBiLSTM` architecture on the combined set.
   - Validate on a held-out subset of MIR-ST500 + the full Vocadito set.
2. Compare to the F-2 Vocadito-only baseline (mean F1 0.4652).
3. If MIR-ST500 pretrain helps (≥ 0.55 fold-mean F1), fine-tune on
   Vocadito and re-test there.
4. If the pretrained model passes the Vocadito Phase E item 7 criterion
   (≥ +0.01 MV2H), promote as the production offset path in
   `humscribe.pitch.voicing`.

## Estimated compute

- Training: ~30 min CPU (84 songs × 10s × 30 epochs × 5 folds).
  Could run on CPU because the model is small (~95k params).
- GPU-free, so it can co-run with anything else.

## Files

- Audio: `/workspace/datasets/mirst500/audio_partial/<id>.mp3`
- Features: `/workspace/.cache/mirst500_formant/<id>.npz`
- Labels: `/workspace/datasets/mirst500/repo/MIR-ST500_20210206/MIR-ST500_corrected.json`
- `scripts/prep_mirst500_partial.py`
- `scripts/prep_mirst500_formant.py`
- `humscribe/train/formant_offset.py` (the model arch — reused from F-2)
