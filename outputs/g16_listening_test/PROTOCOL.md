# G-16 — C5b LoRA listening test protocol (10-pair set complete)

Phase G item G-16 closes the v3 spec item 5 unverifiable criterion
("subjective melody-following ≥ 3.5/5") by producing the eval artifacts
(audio pairs + Google Form template) that humans can run.

## What's shipped

10 audio pair sets ready for listening, located in
`outputs/c5_vs_c5b_multi/`:

| pair # | source MIDI | base WAV | c5b WAV |
|---|---|---|---|
| 1 | JSB Chorale BWV 85.6 | bwv85.6_base.wav | bwv85.6_c5b_r64.wav |
| 2 | JSB Chorale BWV 86.6 | bwv86.6_base.wav | bwv86.6_c5b_r64.wav |
| 3 | JSB Chorale BWV 87.7 | bwv87.7_base.wav | bwv87.7_c5b_r64.wav |
| 4 | JSB Chorale BWV 88.7 | bwv88.7_base.wav | bwv88.7_c5b_r64.wav |
| 5 | JSB Chorale BWV 89.6 | bwv89.6_base.wav | bwv89.6_c5b_r64.wav |
| 6 | JSB Chorale BWV 90.5 | bwv90.5_base.wav | bwv90.5_c5b_r64.wav |
| 7 | JSB Chorale BWV 91.6 | bwv91.6_base.wav | bwv91.6_c5b_r64.wav |
| 8 | JSB Chorale BWV 94.8 | bwv94.8_base.wav | bwv94.8_c5b_r64.wav |
| 9 | JSB Chorale BWV 96.6 | bwv96.6_base.wav | bwv96.6_c5b_r64.wav |
| 10 | JSB Chorale BWV 101.7 | bwv101.7_base.wav | bwv101.7_c5b_r64.wav |

All 10 pairs rendered with MusicGen-Melody-Large 3.3B + C5b r=64 LoRA
adapter (`checkpoints/musicgen_lora_c5_jsb/step_1500`), prompt "bach
four-part chorale played on church organ", duration 10 s, seed 0.

## Listening instructions for raters

For each pair, raters hear the source MELODY clip (the chorale soprano)
first, then two arrangement variants labelled "A" and "B" in random
order (A/B labels set per-rater).

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

Pass: mean rating for C5B across 5 raters × 10 pairs ≥ 3.5/5. Stretch:
C5B mean - BASE mean ≥ 0.5 (the LoRA must measurably help).

## Mechanism evidence (chroma similarity, agent-side)

The chroma similarity between input melody and arrangement (pre-human-eval
sanity check):

| pair | ref_sim | base_sim | c5b_sim |
|---|---|---|---|
| BWV 90.5 | 0.490 | 0.561 | **0.730** |
| BWV 91.6 | 0.500 | 0.539 | **0.696** |
| BWV 94.8 | (see _item-g16_extra_pairs.json) | | |
| BWV 96.6 | (see _item-g16_extra_pairs.json) | | |
| BWV 101.7 | (see _item-g16_extra_pairs.json) | | |

(BWV 85.6–89.6 sim values are in the prior session's `_phase_f_F4_c5_vs_c5b_multi.json`.)

C5b consistently yields higher chroma similarity to the source melody
than the base model on these chorales — supports the hypothesis that
the human-rated "follows the melody" question will favour the C5b variant.

## Google Form template

A ready-to-fill Form spec is in `outputs/g16_listening_test/google_form.md`.
The form embeds the audio files via Drive links (raters must have Drive
access to play; alternative direct-upload paths are documented there).

## Submission CSV path (non-Google-Form)

`outputs/g16_listening_test/ratings.csv` with header
`rater,pair,variant,rating`. Variant = "BASE" or "C5B".

After 5 raters complete: aggregate mean(C5B) and mean(BASE). Pass: mean(C5B) ≥ 3.5/5.
