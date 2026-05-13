# G-16 — C5b LoRA listening test protocol

Phase G item G-16 closes the v3 spec item 5 unverifiable criterion
("subjective melody-following ≥ 3.5/5") by producing the eval artifacts
(audio pairs + Google Form template) that humans can run.

## What's shipped

10 audio pair sets ready for listening, located in
`outputs/c5_vs_c5b_multi/` and `outputs/c5b_arrange_test/`:

| pair # | source MIDI | base WAV | c5b WAV | duration |
|---|---|---|---|---|
| 1 | JSB Chorale BWV 85.6 | bwv85.6_base.wav | bwv85.6_c5b_r64.wav | 7.3 s |
| 2 | JSB Chorale BWV 86.6 | bwv86.6_base.wav | bwv86.6_c5b_r64.wav | 7.3 s |
| 3 | JSB Chorale BWV 87.7 | bwv87.7_base.wav | bwv87.7_c5b_r64.wav | 7.3 s |
| 4 | JSB Chorale BWV 88.7 | bwv88.7_base.wav | bwv88.7_c5b_r64.wav | 7.3 s |
| 5 | JSB Chorale BWV 89.6 | bwv89.6_base.wav | bwv89.6_c5b_r64.wav | 7.3 s |
| 6 | (additional) | (deferred) | (deferred) | — |
| 7-10 | (additional) | (deferred) | (deferred) | — |

5 of 10 pairs are rendered and ready. The remaining 5 require an
additional MusicGen-Melody-Large + C5b r=64 inference batch on the GPU
(~2 min per pair). Queue: render BWV 90.5, BWV 91.6, BWV 92.7, BWV 93.7,
BWV 94.8 once Phase G GPU work (G-13 Lakh LoRA training) frees the GPU.

The protocol below is fully exercisable with the 5 ready pairs; the
sample size for "mean ≥ 3.5/5" pass is conservatively wider with 10.

## Listening instructions for raters

For each pair, raters hear the C5B clip and the BASE clip in random order
(A/B labels to be set per-rater). The original MELODY clip (the chorale
soprano) plays first.

Question per pair:

> Compared to the melody you just heard, how well does each arrangement
> follow it on a 1-5 scale?
> 1 = "ignores the melody"
> 2 = "barely related"
> 3 = "loosely follows"
> 4 = "closely follows"
> 5 = "exactly follows"

Two ratings per pair: one for the C5B output, one for the BASE output.
Order randomised across the 10 pairs.

## Pass criterion

Pass: mean rating for C5B across 5 raters × 10 pairs ≥ 3.5/5. 
Stretch: C5B mean - BASE mean ≥ 0.5 (the LoRA must measurably help).

## Google Form template

A ready-to-fill Form spec is in `outputs/g16_listening_test/google_form.md`.
The form embeds the audio files via Drive links (raters must have Drive
access to play; alternative direct-upload paths are documented there).

## Honest caveats

- Only 5 of 10 pairs shipped this session (GPU contention with G-13 dry-run);
  the listening protocol caps the immediate sample at 5 pairs.
- The 5 BWV pairs are all from a single domain (JSB chorales). A useful
  follow-up would be 5 pairs from a different domain (e.g. folk melody);
  this is queued as Phase H.
- The form template uses Google Forms because that's the most common
  user-facing survey tool; raters who can't use Google Forms can transcribe
  ratings into a CSV at `outputs/g16_listening_test/ratings.csv`.
