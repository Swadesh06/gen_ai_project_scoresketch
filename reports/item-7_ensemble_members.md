# item-7 — music-theory ensemble members (interim)

## Goal

Phase E item 7 per `task_description_v3.md`. Integrate music-theory-guided
ensemble members targeting specific bias sources in the pipeline.

## Status

| ME | description | status | result |
|---|---|---|---|
| ME-1 | pYIN diversifier (librosa, CPU) | implemented | eval deferred (GPU contention with C5 LoRA) |
| ME-4 | tonal-meter prior on DP | implemented + evaluated | **discard** at λ=2: −0.006 mean MV2H on 5 ASAP |
| ME-7 | anacrusis detection | implemented + validated | **conservative; no false positives**; no clear win case in eval set |
| ME-9 | line-of-fifths spelling | implemented + evaluated | **discard** at current tuning: pitches preserved ✓ but accidentals don't drop (mean −4.6%) |
| ME-10 | meter-template ensemble | not started | — |
| ME-11 | formant-band onset detector | not started | — |
| ME-14 | MV2H system-level ensemble | not started | depends on item 6 outputs |

ME-3 (SwiftF0), ME-6 (chord recognition), ME-13 (voice legality) are
deferred per v3 spec (incremental / sub-research-project / B76 near ceiling).

## Per-member detail

### ME-9 — line-of-fifths spelling

`reports/exp_ME9_spelling.md`. Pitches preserved on all 5 ASAP pieces
(correctness ✓), but accidentals increased on Chopin Berceuse (96→116,
+20.8%) because KrumhanslSchmuckler picked the wrong key on the first
30 s of the piece. Net: kept behind flag, not promoted.

### ME-4 — tonal-meter prior

`reports/exp_ME4_tonal_prior.md`. Built P(scale_degree | beat_position)
from music21's Bach chorale iterator (200 pieces, 46k notes). λ=2.0 post-
DP-shift produced −0.006 mean MV2H on 5 ASAP pieces. The prior moves
already-correct YMT3 timings to fit Bach-chorale tonality, which is the
wrong direction on clean inputs. Phase F retry: humming-side eval +
lower λ.

### ME-7 — anacrusis detection

`reports/_exp_ME7_anacrusis.json`. Working detector with conservative
threshold: first-note duration < 0.6 × mean of next 4 notes AND landing
within 300 ms of beat 0. All 5 ASAP pieces correctly returned no pickup
(no false positives). No piece in the eval set has a classic anacrusis
pattern, so we can't demonstrate a beat-shift gain here. The detector is
ready; integration into the pipeline is a one-line `if shift_for_pickup(...)`
when beats are extracted.

### ME-1 — pYIN diversifier

`humscribe/ensemble/me1_pyin.py`. Evaluator built (`eval_me1_pyin.py`)
but blocked on GPU contention — JSB LoRA training is using 14.6 GB and
PESTO/CREPE can't fit alongside. Will run after C5 finishes.

## Decision

None of the implemented members has cleared the +0.01 MV2H pass criterion
on ASAP. The pattern is consistent: simple music-theory priors don't beat
already-good YMT3 transcription on clean instrument audio. The Phase F
candidates are:

1. **Re-run ME-4 on Vocadito** with a humming-tuned prior + lower λ.
2. **ME-11 formant-band onset detector** for the Vocadito offset20 gap.
3. **ME-14 MV2H system-level ensemble** — depends on item 6 finishing.
4. **ME-1 ensemble verification** once GPU is free.

## Files

- `humscribe/ensemble/__init__.py`
- `humscribe/ensemble/me1_pyin.py`
- `humscribe/ensemble/me4_tonal_meter_prior.py`
- `humscribe/ensemble/me7_anacrusis.py`
- `humscribe/ensemble/me9_line_of_fifths.py`
- `scripts/eval_me{1,4,7,9}_*.py`
- `reports/exp_ME{4,9}*.md`
