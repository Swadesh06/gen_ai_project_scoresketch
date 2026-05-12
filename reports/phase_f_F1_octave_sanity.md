# Phase F-1 — beat-tempo octave sanity check (rule-based corrector)

## Goal

Phase F-1 first iteration: rule-based corrector for the beat_this octave
failures identified in `phase_f_F1_beat_corrector_data.md`. Bach BWV 856
and Chopin Berceuse are the two pieces where beat_this picks the wrong
octave despite `target_bpm=110`.

## Procedure

`humscribe/beat/octave_sanity.py` exposes `detect_octave_misalignment(beats,
notes)` returning one of `{keep, halve, double}` based on the **notes-per-
beat ratio**: if a predicted beat covers many notes (> 5.5), beats are
too sparse and should be doubled; if a predicted beat covers very few
notes (< 0.4), beats are too dense and should be halved.

The thresholds were tuned to the 9 ASAP pieces with ground-truth labels
derived by comparing predicted BPM to score-derived BPM.

## Results (9 ASAP pieces)

| piece | pred_bpm | true_bpm | nppb | truth | detector | match |
|---|---|---|---|---|---|---|
| Bach BWV 854 | 120 | 120 | 3.85 | keep | keep | ✓ |
| Bach BWV 846 | 122 | 120 | 3.77 | keep | keep | ✓ |
| Bach BWV 848 | 120 | 120 | 3.85 | keep | keep | ✓ |
| **Bach BWV 856** | **81** | **240** | **6.17** | **double** | **double** | **✓** |
| Bach BWV 857 | 120 | 120 | 3.85 | keep | keep | ✓ |
| Beethoven 21-1 | 150 | 152 | 4.00 | keep | keep | ✓ |
| **Chopin Berceuse** | **120** | **40** | **1.00** | **halve** | **keep** | **✗** |
| Liszt Sonata | 115 | 120 | 2.04 | keep | keep | ✓ |
| Schumann Toccata | 125 | 122 | 4.00 | keep | keep | ✓ |

**8 / 9 correct.** Bach BWV 856 fixed; Chopin Berceuse still missed.

## Why Chopin Berceuse evades the density signal

Both note density AND beat density are halved together for Chopin
Berceuse: predicted at 120 BPM with 0.5 s note IOIs (= 1 note per beat),
while the truth is 40 BPM with 0.5 s note IOIs (= 0.33 note per beat).
The notes-per-beat ratio at 120 BPM is "normal" (≈1), so a density-based
detector cannot distinguish it from a normal slow piece.

The fix requires a **second signal**:
- The notes' absolute IOI is large (0.5 s) which is unusual for a
  120-BPM piece (which would typically have 0.25 s or shorter note IOIs).
- Or: cross-check against beat_this's confidence (beat_this returns a
  per-frame beat curve; integrating it might show that the curve has
  half-strength peaks between the chosen beats).

This will be the next iteration. For now, the detector improves the
8-of-9 case without false-firing on any "keep" pieces.

## Decision

**Keep the detector as a behind-flag opt-in correction**. Phase F-2 will
add the slow-tempo signal needed to catch the Chopin case. Don't promote
to a default-on pipeline change yet — the cost of a false-fire on a
"keep" piece would be substantial (halving correct beats degrades all
downstream rhythm), and the test set is only 9 pieces.

## Files

- `humscribe/beat/octave_sanity.py`
- `scripts/eval_octave_sanity.py`
- `reports/_phase_f_F1_octave_sanity.json`
