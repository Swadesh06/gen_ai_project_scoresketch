# exp_B51 — Vocadito A1↔A2 inter-annotator agreement

## Goal
Establish the human ceiling on Vocadito. We've been pushing toward 0.665 (A1) and 0.630 (A2) and hit a stable optimum after 50+ experiments. Need to know if we're at the supervised ceiling or if there's room.

## Procedure
Compute mir_eval transcription F1 for A1 vs A2 ground truth on every Vocadito clip
(40 clips have both A1 and A2 annotations). Onset tolerance 50 ms, pitch tolerance 50 cents,
no offset constraint. Symmetric (A1->A2 == A2->A1).

## Results

**Overall IAA mean F1 = 0.740 ± 0.091**  (40 clips)
- Min: 0.571 (vocadito_35)
- Max: 0.935 (vocadito_8)

| metric | value |
|---|---|
| IAA mean F1 | **0.740** |
| Our A1 F1 (B36b hybrid) | 0.665 |
| Our A2 F1 (B36b hybrid) | 0.630 |
| Gap to ceiling (A1) | -7.5pp |
| Gap to ceiling (A2) | -11.0pp |

## Interpretation

The IAA ceiling is 0.740. Our pipeline is 7.5–11pp below this. A few takeaways:

1. **There is real room to push**, not at ceiling yet. Pre-trained features (MERT, HuBERT)
   or smarter onset detection could close 2–5pp of the gap.
2. **Aiming above 0.74 is over-fitting** to a single annotator's idiosyncratic choices
   (note splitting at slurs, vibrato boundaries, breath marks).
3. **Per-clip variance is huge** (sd=0.091, range 0.36). Clips that are unambiguous
   (vocadito_8 at 0.935) hit IAA above 0.9; ambiguous clips (vocadito_35 at 0.571)
   IAA is below 0.6.
4. **A2 has fewer notes than A1** systematically — A2 splits less aggressively
   (e.g. clip 13: 74 vs 52, clip 21: 97 vs 62). Our pipeline tends toward A1's split style
   (more notes), which is why A1 F1 > A2 F1.

## Plan based on this finding

- B52: switch to **soft IAA scoring** — average F1 across both annotators per clip — gives a
  single robust number per clip.
- B53: **MERT or HuBERT features** for the segmenter (target the gap between 0.665 and 0.74).
- B54: consider a **multi-target loss** that predicts an A1-style and A2-style splitting
  preference, picks closer one per clip.

The 0.665 → 0.740 gap is real and worth attempting. Beyond 0.740 is not meaningful.
