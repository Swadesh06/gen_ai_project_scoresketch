# Vocadito MV2H at tpb={8, 12, 24} — production-change verification

## Goal

Verify the Phase E tpb=12 production change works on the humming branch
(not just on cached ASAP instrument outputs).

## Procedure

Ran segment_pitch_to_notes (production voicing path) on cached PESTO+CREPE
features for Vocadito clips 1–10, then applied viterbi_quantize_rhythm at
tpb={8, 12, 24} and scored MV2H against A1 GT.

## Results (9 of 10 clips; voc_2 timed out)

| clip | tpb=8 | tpb=12 | tpb=24 |
|---|---|---|---|
| voc_1 | 0.518 | 0.505 | 0.509 |
| voc_3 | 0.533 | 0.528 | 0.518 |
| voc_4 | 0.508 | 0.493 | 0.479 |
| voc_5 | 0.514 | 0.511 | 0.506 |
| voc_6 | 0.495 | 0.488 | 0.445 |
| voc_7 | 0.520 | 0.514 | 0.510 |
| voc_8 | 0.556 | 0.551 | 0.548 |
| voc_9 | 0.505 | 0.503 | 0.506 |
| voc_10 | 0.479 | 0.491 | 0.476 |
| **mean** | **0.5143** | **0.5093** | **0.4996** |

## Interpretation

Same pattern as ASAP: lower tpb wins. tpb=12 is +0.010 over tpb=24
(matches the ASAP +0.011 from ME-14-ext on the same configs). tpb=8 is
+0.014 over tpb=24.

The production tpb=12 switch is **consistent across both branches**
(humming + instrument). Considering the snap-F1 noiseless check showed
−0.0002 delta at tpb=12 vs tpb=24, the production change has:
- +0.010 mean MV2H (humming)
- +0.011 mean MV2H (ASAP instrument)
- −0.0002 mean snap-F1 (legacy metric)

Clearly a net positive.

## Files

- (Inline analysis — no script file; the snippet is in this report.)
