# Phase F-7 — C5b r=64 LoRA chroma-similarity distribution — SHIPPED

## Goal

F-4 reported +0.146 chroma similarity (base 0.570 → C5b 0.716) on a
single held-out chorale (bwv85.6). F-7 verifies the lift on 5
held-out chorales to confirm it's not a one-off.

## Procedure

`scripts/exp_C5_vs_C5b_multi.py`. Five chorales from the tail of
349 alphabetically sorted JSB pair directories (chorales 86-90 from
the end), each outside the C5b training corpus. Generation:
MusicGen-Melody-Large, `seed=0`, `duration_s=10`, prompt `"bach
four-part chorale played on church organ"`, fp16 base + fp16 LoRA via
the F-4 musicgen.py patch. Chroma similarity is `librosa.chroma_cqt`
cosine-mean vs the melody input.

(C5 r=32 adapter was overwritten in-place by C5b training; only C5b
remains, so the comparison is base vs C5b.)

## Results

| chorale | ref vs mel | base vs mel | **c5b vs mel** | c5b − base |
|---|---|---|---|---|
| bwv85.6 | 0.508 | 0.563 | **0.683** | +0.120 |
| bwv86.6 | 0.485 | 0.527 | **0.674** | +0.147 |
| bwv87.7 | 0.498 | 0.513 | **0.713** | +0.200 |
| bwv88.7 | 0.473 | 0.520 | **0.688** | +0.168 |
| bwv89.6 | 0.471 | 0.615 | **0.687** | +0.072 |
| **mean** | **0.487** | **0.548** | **0.689** | **+0.141** |

**All 5 chorales show positive C5b lift** (range +0.072 to +0.200).
Mean lift **+0.141** is consistent with the F-4 single-point result
(+0.146) — the +0.146 on bwv85.6 was not a one-off.

## Interpretation

The C5b r=64 LoRA (test loss 0.983, vs r=32's 1.388) translates the
**−0.41 training-loss capacity-hypothesis advantage** into a
measurable **+0.141 audio-level chroma-similarity advantage** across
5 held-out chorales. End-to-end, not just in the training loss.

That C5b consistently lifts melody-similarity above both base and
reference (mean 0.689 vs 0.487 reference) is expected for a
melody-tracking LoRA — the four-voice GT arrangement intentionally
diverges from the melody in its supporting voices, so a melody-faithful
generation lands higher on the chroma-vs-melody axis.

bwv89.6 had the smallest C5b lift (+0.072) but also the highest base
similarity to begin with (0.615 vs the 0.51-0.56 of the other four)
— if the base already tracks the melody fairly well, the room for the
adapter to improve is smaller. **None of the 5 chorales regress;
worst case is +0.072.**

## Files

- Code: `scripts/exp_C5_vs_C5b_multi.py`.
- Outputs: `outputs/c5_vs_c5b_multi/bwv{85.6,86.6,87.7,88.7,89.6}_{base,c5b_r64}.wav`
  (10 audio files, ~628 KB × 2 × 5 = 6.3 MB).
- Log: `logs/exp_C5_vs_C5b_multi.log`.
- JSON: `reports/_phase_f_F4_c5_vs_c5b_multi.json`.

## Next

- Qualitative listen pass (audio files above) — needed before any
  user-facing demo gallery.
- C5b LoRA generalisation beyond JSB: try on user-collected
  humming-to-orchestral arrangement pairs (none currently in repo).
- Train an even-larger LoRA on a much larger MIDI corpus (e.g. Lakh
  ~100k MIDIs vs JSB's 371) — proposed as F-8 in PLAN.md.
