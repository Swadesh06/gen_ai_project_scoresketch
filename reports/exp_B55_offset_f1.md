# exp_B55 — Vocadito Onset-Offset F1

## Goal
Our headline F1 (0.665) uses `offset_ratio=None` — offset durations are not constrained.
This hides the fact that note durations may be wrong. Re-evaluate Vocadito A1 with
stricter `offset_ratio=0.2` and `0.5` to measure duration quality.

## Procedure
- Run pipeline (mode=soft, hybrid voicing) on all 40 Vocadito A1 clips.
- Compute mir_eval `precision_recall_f1_overlap` with three offset ratios:
  None (current default), 0.2 (strict), 0.5 (loose).
- `offset_min_tolerance=0.05` to allow small absolute slack.

## Results

| metric | F1 | gap |
|---|---|---|
| f_no_offset (current default) | **0.665** | — |
| f_offset50 (loose) | 0.573 | -9.2pp |
| f_offset20 (strict) | **0.439** | **-22.6pp** |

## Interpretation

The 22.6pp drop between no-offset and offset20 F1 means our pipeline's note durations
are wrong on ~half the matched notes. The onset is right, the pitch is right, but
the offset is off by >20% of the GT duration.

Why? Two related causes:
1. **Voicing-thresholded segmentation** ends a note when voicing dips below `vt`. But
   vibrato + breath noise can cause spurious dips, ending notes too early.
2. **min_note_seconds=0.052** filters very short notes, but doesn't extend short notes
   to musically-plausible lengths. So a 70ms note from a glottal-stop misfire stays 70ms.

## Implications

We've been optimizing on no-offset F1 (the spec's chosen metric), and our +12.7pp gain
since Phase A is real on that metric. But adding offset constraints reveals the
duration-estimation step is much weaker than the onset-detection step.

**Two paths**:
1. **Defend the current optimization target** — no-offset F1 is what mir_eval defaults
   to in monophonic transcription literature. Don't change.
2. **Add a duration-quantization step** — snap note durations to musically-plausible
   subdivisions of the inferred beat (1/8, 1/4, dot-1/8, etc.) using DP. This is what
   we already do for ASAP. Bring it to Vocadito.

## Next
- B56: add Cemgil-Kappen DP duration quantization to the Vocadito pipeline. Target:
  push offset20 F1 from 0.439 → 0.55+, ideally without sacrificing no-offset 0.665.
  This is the duration analog of what voice tracking did for ASAP rhythm.
