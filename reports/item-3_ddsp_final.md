# item-3 — DDSP humming→instrument experiment — FINAL

## Goal

v3 spec item 3. Route hummed audio through Magenta's DDSP timbre
transfer (violin), then transcribe the resulting violin audio through
the humming pipeline. Compare:
- direct: humming pipeline on raw hum
- ddsp: humming pipeline on DDSP-transferred violin audio
- ensemble: union of direct + non-overlapping DDSP notes

## DDSP install (formerly blocked)

Earlier session noted that DDSP imports failed after 6 `--no-deps`
installs because of an open dep chain. This session resolved it by
installing the remaining ~15 packages:

```
tensorflow-metadata etils toml note_seq pydub mido absl-py
bokeh tornado xyzservices ipython traitlets prompt_toolkit
pygments decorator pickleshare wcwidth executing asttokens
pure_eval stack_data jedi parso attrs intervaltree
importlib_resources promise
```

`from ddsp.training import inference` now succeeds (the
`tensorflow_metadata` ImportError surfaces from a deferred import
in `tensorflow_datasets` but doesn't block the autoencoder).

## Wrapper + checkpoint

`humscribe/pitch/timbre_transfer/ddsp_violin.py` —
solo_violin_ckpt loaded from
`/workspace/.cache/ddsp_checkpoints/solo_violin_ckpt/`
(downloaded from `gs://ddsp/models/timbre_transfer_colab/2021-07-08/`,
58 MB total). Loads `ckpt-40000.{data,index}` + `operative_config-0.gin` +
`dataset_statistics.pkl`. `transfer(audio, sr)` chunks audio into 4-s
windows (the canonical DDSP demo length), runs the autoencoder, and
concatenates outputs at 16 kHz. CPU-only (TF can't see the GPU due to
PyTorch/TF CUDA registry conflict; ~27 s of wall per 4 s of audio).

## Pipeline + eval

`scripts/eval_item3_ddsp_ensemble.py` runs all 40 Vocadito A1 clips
through 3 configurations. Outputs cached at
`/workspace/.cache/ddsp_violin_vocadito/vocadito_{1..40}.wav`.

Wall: **185 minutes** to process all 40 clips CPU-only.

## Results — 40 Vocadito A1 clips

| config | mean noff F1 | range | v3 criterion | pass? |
|---|---|---|---|---|
| **direct** (production humming pipeline) | **0.6181** | 0.45–0.80 | ≥ 0.55 | **✓ PASS** |
| ddsp (DDSP→violin→pipeline) | 0.1381 | 0.00–0.35 | — | n/a |
| **ensemble** (direct + ddsp dedup) | **0.4835** | 0.35–0.64 | ≥ 0.71 | **✗ FAIL** |

**DDSP path produces zero matching notes on 14 of 40 clips** and ≤ 0.30
on a further 22. The ensemble underperforms direct on every clip
because the DDSP false-positives drag precision down without lifting
recall.

## Why DDSP didn't help

The DDSP solo_violin autoencoder reconstructs violin spectral
characteristics from input f0+loudness features. In its native domain
(monophonic instrumental recordings) it preserves pitch well. On
**hummed audio**, three failure modes appear:

1. **Pitch normalisation against dataset_statistics**: the wrapper
   loudness-shifts inputs to match training-set loudness. For hummed
   input this can land on the wrong octave.
2. **Chunked transfer boundaries**: 4-second windows are stitched
   directly. PESTO/CREPE detect spurious onsets at the boundary
   transients (no cross-fade applied).
3. **Vibrato + breathy attacks**: vocal vibrato (~5 Hz LFO on f0) gets
   amplified by the autoencoder's preprocessor; the resulting violin
   audio has unstable pitch traces that fail PESTO confidence
   thresholds.

A better wrapper could mitigate (2) with overlap-add and (1) by
disabling the loudness shift. (3) is structural — the solo-violin
checkpoint wasn't trained on vibrato-heavy input. A different
checkpoint (e.g. flute, less vibrato-sensitive) might help.

## v3 item 3 strict pass-criterion result

- Direct DDSP path Voc A1 ≥ 0.55: **✓ passed (0.618 > 0.55)**
  — but this is the production humming pipeline, not anything DDSP
  contributed. Pass is by-default since DDSP wasn't actually used in
  the direct config.
- Ensemble path Voc A1 ≥ 0.71: **✗ failed (0.484 vs 0.71, gap −0.226)**
  — DDSP path is too noisy to ensemble with direct.

## Files

- `humscribe/pitch/timbre_transfer/__init__.py`
- `humscribe/pitch/timbre_transfer/ddsp_violin.py`
- `scripts/eval_item3_ddsp_ensemble.py`
- `/workspace/.cache/ddsp_checkpoints/solo_violin_ckpt/` — Magenta checkpoint
- `/workspace/.cache/ddsp_violin_vocadito/vocadito_{1..40}.wav` — cached transfers
- `reports/_phase_e_item3_ddsp_ensemble.json` — per-clip + means

## Future directions (Phase F)

- Try the DDSP `solo_flute2` checkpoint (less vibrato-sensitive than violin)
- Cross-fade chunk boundaries to remove transient onsets
- Disable loudness normalisation (preserve original loudness curve)
- Or: use a different hum→instrument model entirely
  (e.g. RVC voice-conversion, Seamless audio-to-audio).
