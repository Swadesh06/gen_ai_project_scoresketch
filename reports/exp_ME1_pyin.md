# ME-1 — pYIN diversifier — discard at current weighting

## Goal

Phase E item 7 ME-1 per `task_description_v3.md`. Use librosa's pYIN
(pure DSP, no neural net) as an uncorrelated pitch+voicing vote
alongside PESTO and CREPE-periodicity for the humming branch. Different
failure modes from neural trackers.

## Procedure

- `humscribe/ensemble/me1_pyin.py`: `track_pitch_pyin` returns
  (times, hz, voicing) at 10 ms hop. `vote_with_pesto_crepe` resamples
  pYIN onto the PESTO time grid, then:
  - **Agreement** (|pitch_pesto - pitch_pyin| < 50 cents): voicing is
    boosted to max(pesto, pyin) — confidence increases.
  - **Disagreement**: pesto's pitch wins, voicing dampened ×0.7.
- `scripts/eval_me1_pyin.py` compares MV2H of (baseline) PESTO+CREPE
  voicing-segmented + DP against (ME-1) pYIN-voted voicing on 10 Vocadito
  clips. Both versions pass through the same DP at TPB=24.

## Results

| clip | base MV2H | ME-1 MV2H | Δ |
|---|---|---|---|
| voc_1 | 0.5097 | 0.5119 | +0.0022 |
| voc_2 | (timed out — MV2H DTW failed) | — | — |
| voc_3 | 0.5175 | 0.5096 | −0.0079 |
| voc_4 | 0.4785 | 0.4710 | −0.0076 |
| voc_5 | 0.5046 | 0.4841 | **−0.0205** |
| voc_6 | 0.4445 | 0.4251 | −0.0194 |
| voc_7 | 0.5048 | 0.5156 | +0.0108 |
| (8-10 — eval killed for GPU contention with C5b) | | | |

Mean across 6 succeeded clips: **−0.0071** (net negative).

## Interpretation

The ×0.7 voicing damping on disagreement is too aggressive — it removes
real notes more often than it suppresses false positives. The
disagreement cases are typically borderline notes where PESTO and pYIN
both have meaningful confidence; dampening voicing causes a real note
to fall below the segmenter's voicing_threshold.

The agreement case (boost to max) helps marginally (voc_1, voc_7 small
positive deltas) but it's swamped by the disagreement cost.

## Decision

**Discard at this weighting**. Phase F retry candidates:
1. Use disagreement as a *signal* (not a damping factor): widen the
   voicing search window when PESTO and pYIN diverge, instead of
   suppressing.
2. Train a small classifier on (PESTO_vc, pYIN_vc, agreement, pitch_diff)
   → ground-truth voicing label using Vocadito annotations. 40 clips is
   small but might be enough for a 4-feature -> 1-output logistic
   regression.
3. Use pYIN as a **second pitch trace** (not a voicing modifier) and
   ensemble at the note level via the existing `humscribe.pitch.ensemble`
   path.

## Files

- `humscribe/ensemble/me1_pyin.py`
- `scripts/eval_me1_pyin.py`
- `logs/eval_me1.log` (partial; eval was killed mid-flight due to GPU
  contention with C5b LoRA training — 6 of 10 clips evaluated, sufficient
  to call the negative direction)
