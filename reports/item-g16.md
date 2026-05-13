# item-g16 — C5b LoRA subjective listening artifact

## Goal
task_description_v4.md item G-16. Close the v3 spec item 5 unverifiable criterion ("subjective melody-following ≥ 3.5/5") by producing the listening eval artifacts (10 audio pairs + Google Form template + listening protocol). The agent does NOT run the human eval; the spec assigns the rating itself to human listeners.

## Procedure
- Listening test directory at `outputs/g16_listening_test/`:
  - `PROTOCOL.md` — listening instructions, randomisation rules, pass criterion.
  - `google_form.md` — copy-paste Google Forms spec with per-pair Likert template.
  - `ratings.csv` — empty CSV scaffold for non-Google-Forms submission.
- Audio pair set at `outputs/c5_vs_c5b_multi/`: 10 (melody, base, c5b) sets:
  - Pre-existing 5: BWV 85.6, 86.6, 87.7, 88.7, 89.6 (rendered in prior session).
  - **New 5 added in this strict-measurement run**: BWV 90.5, 91.6, 94.8, 96.6, 101.7 (rendered via `scripts/g16_render_5more.py`, MusicGen-Melody-Large 3.3B + C5b r=64 LoRA adapter step 1500, duration 10 s, seed 0).

## Results

### 10/10 pair set complete

| pair # | chorale | base_sim | c5b_sim | ref_sim |
|---|---|---|---|---|
| 1 | BWV 85.6 | (prior) | (prior) | (prior) |
| 2 | BWV 86.6 | (prior) | (prior) | (prior) |
| 3 | BWV 87.7 | (prior) | (prior) | (prior) |
| 4 | BWV 88.7 | (prior) | (prior) | (prior) |
| 5 | BWV 89.6 | (prior) | (prior) | (prior) |
| 6 | BWV 90.5 | 0.561 | **0.730** | 0.490 |
| 7 | BWV 91.6 | 0.539 | **0.696** | 0.500 |
| 8 | BWV 94.8 | 0.601 | **0.716** | 0.499 |
| 9 | BWV 96.6 | 0.509 | **0.708** | 0.484 |
| 10 | BWV 101.7 | 0.542 | **0.688** | 0.470 |
| **mean (new 5)** | — | **0.5504** | **0.7076** | 0.4886 |

The 5 new pairs all show c5b > base on chroma similarity to the input melody (Δ +0.1572 mean) and c5b > ref (the original arrangement) too (the LoRA over-fits to "follow the soprano line", consistent with its training objective). This is *mechanism evidence*; the human-rated "follows the melody" question should favour c5b but is the actual strict criterion that the spec defers to humans.

### Honest mean-rating status
The strict criterion `mean rating ≥ 3.5/5` requires 5 human raters submitting through the form / CSV path. No human-rater data has been collected this session. The artifact is shipped and exercisable.

## Pass / discard
- **10 pairs shipped**: original 10, observed **10** → **passed-with-metric-evidence**.
- **Protocol + Form shipped**: ✓.
- **Mean rating ≥ 3.5/5**: human-rater dependent, spec-allowed deferral.

**Net G-16 status: ARTIFACT FULLY SHIPPED (10/10 pairs + protocol + form). Per the spec, this is the maximum the agent can close; the rating step is delegated to human listeners.**

## Next
- Distribute Google Form to 5 raters (the form's TODO lines indicate Drive links to fill in).
- Aggregate ratings into `outputs/g16_listening_test/ratings.csv`.
- Re-compute mean rating and flip the JSON `mean_rating_ge_3.5` field once data is in.
