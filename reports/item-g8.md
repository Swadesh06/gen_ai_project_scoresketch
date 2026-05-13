# item-g8 — round-trip self-consistency metric

## Goal
task_description_v4.md item G-8. Reference-free signal: audio → pipeline.transcribe → MIDI → synthesise → MFCC-DTW distance to the original audio. Strict pass: correlation with MV2H |r| ≥ 0.3 on 9 ASAP pieces, successfully flags Liszt (highest round-trip distance), catches ≥ 80% of MV2H < 0.30 cases.

## Procedure
- New module `humscribe/eval/round_trip.py`:
  - `notes_to_pretty_midi(notes, bpm)` builds a single-instrument PrettyMIDI.
  - `round_trip_distance(audio_ref, sr_ref, notes, bpm, synth_fs)` synthesises notes with `PrettyMIDI.synthesize(fs=16k)` (sinusoidal), MFCCs both sides (13 coeffs, hop=512), runs DTW cosine cost, normalises by path length.
- New script `scripts/eval_round_trip.py`: runs the metric on the 9 cached ASAP pieces using the YourMT3+ note cache, computes Pearson + Spearman correlation against the G-1+G-2 MV2H rows.
- CPU only; ~30 s for the full 9-piece set.

## Results

| piece | rt_distance | mv2h | n_notes |
|---|---|---|---|
| Bach__Fugue__bwv_846 | 0.04149 | 0.6252 | 383 |
| Bach__Fugue__bwv_848 | 0.03285 | 0.6534 | 347 |
| Bach__Fugue__bwv_854 | 0.03363 | 0.6733 | 374 |
| Bach__Fugue__bwv_856 | 0.03314 | 0.5801 | 436 |
| Bach__Fugue__bwv_857 | 0.02685 | 0.6486 | 251 |
| Beethoven__Piano_Sonatas__21-1 | 0.02826 | 0.5885 | 501 |
| Schumann__Toccata | 0.03558 | 0.6355 | 722 |
| Chopin__Berceuse_op_57 | 0.02025 | 0.5448 | 131 |
| Liszt__Sonata | 0.01960 | 0.5865 | 142 |

- **Pearson(rt_distance, mv2h) = +0.642** (|r| = 0.642 → meets the strict |r| ≥ 0.3)
- **Liszt has the LOWEST distance (0.020)**, not the highest. Sign is reversed.
- No piece in the 9-piece set has MV2H < 0.30, so the "catches ≥ 80%" criterion is vacuous.

## Interpretation
- Pearson |r| crosses the 0.3 threshold but with the **wrong sign**. Higher round-trip distance correlates with higher (better) MV2H, not lower. This is the opposite of what a quality signal should do.
- Inspection: distance correlates with **note count** (Bach Fugues = 250-440 notes, distance 0.027-0.041; Liszt = 142 notes, distance 0.020). Sinusoidal synthesis of a sparse piece is acoustically closer to the original than a dense Fugue, regardless of correctness. Note count also correlates positively with MV2H on this set (more notes = better polyphonic match coverage in the metric), so distance and MV2H move together for the wrong reason.
- Liszt has the lowest distance (best by the metric) but is the worst-MV2H piece (0.587 — comparable to Chopin Berceuse). The metric does not flag Liszt's structural failure.
- A working round-trip metric for HumScribe needs to either:
  - Subtract a notecount/density baseline before reporting the residual.
  - Use a timbre-invariant chroma or onset-strength distance instead of MFCC.
  - Synthesise with a soundfont matching the source instrument (not sinusoidal).

## Pass / discard
- **|r| ≥ 0.3**: original 0.3, observed 0.642 → PASS on magnitude but the sign is inverted (positive correlation, wrong direction).
- **Successfully flags Liszt (highest distance)**: original "Liszt highest", observed "Liszt lowest" → FAIL.
- **Catches ≥ 80% of MV2H < 0.30**: 0 pieces have MV2H < 0.30 → vacuous.

**Net G-8 status: DISCARDED. The MFCC-DTW round-trip distance correlates with MV2H, but through note-count rather than per-piece quality — Liszt's structural failure is not flagged. The infrastructure is shipped in `humscribe/eval/round_trip.py` so a Phase H follow-up (chroma DTW + density-normalised score) can swap the distance metric without re-plumbing.**

## Next
- Phase H: chroma-distance round-trip + per-density normalisation.
- G-10 bar-level diagnostic gives a stronger Liszt signal (0.490 vs Bach 0.99) and may be a better unsupervised quality proxy.
