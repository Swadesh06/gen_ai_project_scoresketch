# exp_B81 — Anticipatory Music Transformer continuation demo (Phase D)

## Goal
Bolt on a "score continuation" feature using
[stanford-crfm/music-medium-800k](https://huggingface.co/stanford-crfm/music-medium-800k)
(Thickstun et al. 2024). Transcribed humming MIDI → AMT → continuation.

This is a different generative-AI family than MusicGen:
- **MusicGen**: autoregressive over EnCodec audio tokens, generates raw audio
- **AMT**: autoregressive over anticipatory MIDI tokens, generates symbolic music

Pass criterion: end-to-end MIDI → AMT → MIDI without crash. Quality is
qualitative.

## Procedure

`scripts/exp_B81_amt_continuation.py`:
1. Install `anticipation` (Thickstun GitHub package, MIT)
2. Transcribe Vocadito clip 1 with the production pipeline
3. Convert NoteEvent list → PrettyMIDI → AMT events
4. Call `generate(model, start_time=last_note_offset_s, end_time=+15s, inputs=in_events)`
5. Write resulting MIDI

## Results

| metric | value |
|---|---|
| n input notes (transcribed humming) | 69 |
| n input AMT events | 207 |
| n output AMT events | 207 |
| **new events generated** | **0** |
| generation wall | 0.24 s |
| Model params | ~360M (medium-800k = 800k training steps on 360M params) |

**Loaded cleanly, generated nothing.** AMT echoed the input but did not
add new events in the requested 15 s window.

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/cws59xyy

## Interpretation

The AMT model is trained on polyphonic instrumental music (Lakh MIDI,
GiantMIDI-Piano). Vocadito clip 1 is a monophonic vocal melody at low
pitch density (~1.6 notes/sec). AMT's generation distribution likely
predicts "no notes for 15 seconds" with high probability after a sparse
monophonic prompt — the training distribution doesn't include "monophonic
vocal melody continues into more monophonic vocal melody".

Possible Phase E fixes:
1. Use a denser MIDI prompt (e.g. ASAP Bach Fugue first 30 s) — likely
   to generate more aggressively given the in-distribution prompt.
2. Use AMT's "controlled generation" mode where you specify expected
   density / instrument distribution.
3. Try a smaller / larger AMT variant or a different music-LM.

## What's validated
- `anticipation` package (Thickstun) installs from GitHub, uses HF
  transformers `AutoModelForCausalLM`.
- `midi_to_events` and `events_to_midi` round-trip cleanly.
- `generate(model, start_time, end_time, inputs, top_p)` runs to
  completion in ~0.2 s (very fast on this card).
- Output saves as valid MIDI at `outputs/amt_continuation/vocadito_1_continuation.mid`.

## Decision
**Informative.** The infrastructure is in place — `humscribe` can call
AMT in the future with different prompts / settings. As-is, AMT on
monophonic humming generates nothing. A polyphonic test (e.g. transcribed
piano) would be a more honest evaluation of AMT's continuation behavior.

## Status
informative — bolt-on integration validated; quality on monophonic humming
is null. Phase E candidate: test on transcribed piano (Bach BWV 854) to
see if AMT generates meaningful continuations on dense polyphonic prompts.
