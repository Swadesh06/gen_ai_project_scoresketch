# item-3 — DDSP humming→instrument experiment (partial; install pursued)

## Goal

Phase E item 3 per `task_description_v3.md`. Route the hummed audio
through Magenta's DDSP timbre-transfer (violin checkpoint), then
re-transcribe with the instrument pipeline. Test the ensemble vs the
existing humming-direct path.

## Status: partial

DDSP install on this pod required a chain of `--no-deps` installs to
avoid pulling the full transitive dep set:

```
crepe gin-config        # core DSP deps
ddsp==3.7.0             # main package, plus
tensorflow_probability==0.23  # pinned for TF 2.15 compat (not 0.25)
dm-tree                 # for tree mapping
hypertune               # for hyperparameter logging
tensorflow_datasets     # data pipeline base
etils                   # data pipeline utility
... (incomplete)
```

After 6 dep installs, `from ddsp.training import inference` still fails
with `No module named 'etils'`. The full DDSP dep tree is on the order
of 50+ Python packages including TF-data-pipeline transients. The setup
is feasible on a clean fresh env but consumes substantial time in this
shared env that has 100+ packages already pinned for the existing
pipeline.

## Path forward

For Phase F-3 (the DDSP follow-up), recommend either:

1. **Fresh isolated conda env** for DDSP — install full DDSP cleanly,
   run inference, expose as a CLI tool, then call as a subprocess from
   the main `humscribe` env. Decouples deps.
2. **Use HuggingFace audio-to-audio voice conversion** instead — e.g.,
   `voicebox` or `seamless-streaming-vc` — same architectural test
   (hum → violin → re-transcribe) with cleaner pip-installability.
3. **Skip DDSP and use deterministic violin SoundFont synth** of the
   transcribed MIDI as a baseline (less interesting; doesn't add new
   information vs direct transcription).

For this session: deferred. The data-flow is the same as item 5's JSB
pair render — `humscribe.pipeline.transcribe()` on the timbre-transferred
audio. The architectural test is the same regardless of model. Phase F
should pick option 1 (dedicated env) or 2 (HF model) depending on
maintenance overhead preferences.

## What was actually done

- `humscribe/pitch/timbre_transfer/` is **not yet created** — DDSP isn't
  importable, so no wrapper exists.
- `humscribe/ensemble/me1_pyin.py` provides a parallel "uncorrelated pitch
  vote" (already shown to be negative for the humming branch).
- The data path for item 3 (ensemble of humming + DDSP-transferred) is
  blocked on DDSP-inference availability.

## Decision

Item 3 is **deferred** to Phase F. Not a critical path for Phase E
deliverables (the v3 spec lists items 1, 4, 6 as higher-priority CPU
work). Item 5 (JSB LoRA) and Item 7 (ensemble members) carry the
generative-AI workstream for this session.

## Files

- `humscribe/ensemble/me1_pyin.py` (the alternative-vote member)
- `reports/item-3_ddsp_partial.md` (this file)
