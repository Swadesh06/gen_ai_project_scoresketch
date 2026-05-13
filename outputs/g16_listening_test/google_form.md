# G-16 Google Form spec

Use this content to create a Google Form. The form name is
"HumScribe C5b LoRA listening test" and the body has 1 introduction page
plus 10 question pages (one per pair). Per pair we collect a single 1-5
Likert rating per variant (BASE vs C5B), order randomised per rater.

## Introduction page

> This 5-minute test asks you to compare two arrangements of the same
> melody. For each pair you'll hear the original melody (a Bach chorale
> soprano), then two arrangements labelled "A" and "B" in random order.
> Rate how well each arrangement follows the melody on a 1-5 scale.
> 1 = "ignores the melody", 5 = "exactly follows".

## Per-pair page (× 10 pairs)

Embed three audio clips:
1. melody clip (Bach soprano, single instrument)
2. variant A
3. variant B

Two Likert questions:
- "Variant A — how well does it follow the melody?" (1–5)
- "Variant B — how well does it follow the melody?" (1–5)

Per-pair labels (randomised per rater):

| pair | melody (Drive link) | variant A | variant B | A-is-c5b? |
|---|---|---|---|---|
| 1 | (TODO) | bwv85.6_base.wav | bwv85.6_c5b_r64.wav | False |
| 2 | (TODO) | bwv86.6_c5b_r64.wav | bwv86.6_base.wav | True |
| 3 | (TODO) | bwv87.7_base.wav | bwv87.7_c5b_r64.wav | False |
| 4 | (TODO) | bwv88.7_c5b_r64.wav | bwv88.7_base.wav | True |
| 5 | (TODO) | bwv89.6_base.wav | bwv89.6_c5b_r64.wav | False |
| 6-10 | (deferred — see PROTOCOL.md) | | | |

## Submission CSV (alternative path)

If raters can't use Google Forms, paste ratings into
`outputs/g16_listening_test/ratings.csv` with header
`rater,pair,variant,rating`. Variant = "BASE" or "C5B".

After 5 raters complete: aggregate mean(C5B) and mean(BASE). Pass: mean(C5B) ≥ 3.5/5.
