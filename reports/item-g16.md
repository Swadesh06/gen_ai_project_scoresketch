# item-g16 — C5b LoRA subjective listening artifact

## Goal
task_description_v4.md item G-16. Close the v3 spec item 5 unverifiable criterion ("subjective melody-following ≥ 3.5/5") by producing the listening eval artifacts (10 audio pairs + Google Form template + listening protocol). The agent does NOT run the human eval; it ships the artifacts.

## Procedure
- New directory `outputs/g16_listening_test/`:
  - `PROTOCOL.md` — listening instructions, pass criterion, randomisation rules.
  - `google_form.md` — copy-paste-ready Google Forms spec with per-pair question template + Likert wording.
  - `ratings.csv` — empty CSV with header `rater,pair,variant,rating` for the non-Google-Forms submission path.
- Audio assets reused from prior session's `outputs/c5_vs_c5b_multi/` (5 BWV chorale pairs, each base vs c5b_r64 rendered with MusicGen-Melody-Large 3.3B + C5b r=64 LoRA adapter step 1500).
- 5/10 pairs are ready; the remaining 5 require a GPU inference batch (~10 min). Queued as a Phase H follow-up; the protocol caps the immediate sample at 5 pairs to remain exercisable today.

## Results
- 5 audio pairs ready at `outputs/c5_vs_c5b_multi/bwv{85.6,86.6,87.7,88.7,89.6}_{base,c5b_r64}.wav`.
- Listening protocol + Form template + CSV scaffold delivered.
- No human rater data collected this session (the agent doesn't run the survey).

## Pass / discard
- **10 pairs shipped**: 5/10 → **partial-ship-with-deferred-rationale** (5 ready, 5 queued behind G-13 GPU contention).
- **Listening protocol + Form**: → **shipped**.
- **Mean rating ≥ 3.5/5**: → **deferred to human rating** (artifact in place; no rater data yet).

**Net G-16 status: HUMAN-FACING ARTIFACT SHIPPED (partial pair count). The strict ≥ 3.5/5 mean rating cannot pass without human raters; the protocol and 5/10 pair set are ready for that to land.**

## Next
- Render 5 more pair clips (BWV 90.5, 91.6, 92.7, 93.7, 94.8) once the GPU is free of Phase G work; ~10 min on MusicGen-Melody-Large.
- Distribute Google Form to 5 raters; collect responses into `ratings.csv`.
- Recompute mean rating once data is in.
