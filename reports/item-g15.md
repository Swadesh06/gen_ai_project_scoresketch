# item-g15 — DDSP solo_flute2 retest

## Goal
task_description_v4.md item G-15. Replace the failed v3 solo_violin DDSP attempt with three fixes: (a) use `solo_flute2` checkpoint instead of `solo_violin`, (b) crossfade 4 s chunk boundaries with 200 ms overlap-add, (c) disable DDSP's loudness normalisation. Strict pass: direct DDSP→pipeline Vocadito A1 ≥ 0.55, ensemble (direct + DDSP) ≥ 0.65.

## Procedure
- New module `humscribe/pitch/timbre_transfer/ddsp_flute.py` with the three fixes wired in:
  - `_crossfade_concat(chunks, sr, crossfade_ms=200.0)` does linear overlap-add at chunk boundaries — the v3 violin path's audible-click failure mode.
  - `transfer(hum_audio, sr_in)` runs the autoencoder per 4 s chunk and bypasses the loudness-normalisation step (`feats_in` omits the loudness-shift that the violin path used).
  - `is_checkpoint_available()` reports whether `solo_flute2_ckpt` exists at `/workspace/.cache/ddsp_checkpoints/solo_flute2_ckpt/`.
- Checkpoint: the Magenta `solo_flute_ckpt` is at `gs://magenta-data/ddsp/checkpoints/solo_flute_ckpt.zip`. The sandbox does NOT have gsutil and the local cache does NOT have the checkpoint (`/workspace/.cache/ddsp_checkpoints/` contains only `solo_violin_ckpt`).

## Results

### Checkpoint availability
`is_checkpoint_available() == False` on this host. The transfer call raises `FileNotFoundError` with the download instructions baked in.

### Crossfade smoke test
With `solo_violin_ckpt` (the only available checkpoint) wrapped through the new `transfer` path on `vocadito_1.wav`:
- click artefacts at 4 s, 8 s, 12 s boundaries (clearly visible in waveform diff) are eliminated by the 200 ms crossfade.
- audible "loudness pump" at boundaries (the v3 violin failure) is gone.

(The smoke test confirms the new pipeline code is structurally correct; it doesn't substitute for the solo_flute2 evaluation.)

### Vocadito A1 noff F1 with G-15 enabled
Deferred: the `solo_flute2_ckpt` is not on this host. The downstream Vocadito evaluation requires the checkpoint to run.

## Pass / discard
- **Direct DDSP→pipeline Vocadito A1 ≥ 0.55**: original 0.55, observed N/A → **deferred to checkpoint availability**.
- **Ensemble (direct + DDSP) ≥ 0.65**: original 0.65, observed N/A → **deferred**.
- **Three fixes shipped**: crossfade ✓, loudness-norm bypass ✓, solo_flute2 path ✓ → **code shipped**.

**Net G-15 status: CODE SHIPPED, EVALUATION DEFERRED to solo_flute2_ckpt download. The three fixes named in the task description are all in the new module; the integration path is identical to the existing solo_violin module (drop-in replacement). The Vocadito-side measurement will run once the checkpoint lands.**

## Next
- Phase H: download solo_flute2_ckpt (`gsutil cp gs://magenta-data/ddsp/checkpoints/solo_flute_ckpt.zip /workspace/.cache/ddsp_checkpoints/` + unzip), then re-run `scripts/eval_item3_ddsp_ensemble.py` with the flute backend.
