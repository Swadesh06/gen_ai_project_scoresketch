# exp_B7_mtg_qbh_rebaseline — MTG-QBH visual gate, B2-tuned defaults

## Goal
After updating soft-mode defaults from the B2 sweep (vt=0.315, mns=0.052, psw=11, oms=0.026), confirm MTG-QBH visual gate still passes and document any qualitative changes in note counts vs the Phase-A baseline.

## Procedure
- `scripts/gate_mtg_qbh_visual.py --modes soft --n-clips 10`. Same 10 MTG-QBH clips (q1, q10, q100..q107).
- Same pass criterion (>=80% clips with >=1 note).

## Results

| clip | bpm | notes (Phase A soft, old defaults) | notes (B2 soft, new defaults) |
|---|---|---|---|
| q1 | 100.0 | 39 | 35 |
| q10 | 85.7 | 39 | 35 |
| q100 | 125.0 | 46 | 52 |
| q101 | 62.5 | 83 | 74 |
| q102 | 48.0 | 107 | 97 |
| q103 | 136.4 | 47 | 43 |
| q104 | 103.4 | 48 | 46 |
| q105 | 130.4 | 89 | 86 |
| q106 | 78.9 | 81 | 80 |
| q107 | 100.0 | 127 | 121 |

10/10 clips produced >= 1 note. Gate PASS.

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/iye9mret

## Interpretation
Note counts moved by ±5% on average — a small net reduction. The new `psw=11` (vs old 7) smooths over more of PESTO's frame-level pitch wobble, so fewer spurious "different note" decisions; the new `oms=0.026` (vs old 0.08) un-glues notes that were merged before. These two effects partly cancel; net is slightly cleaner segmentation that fragments fewer sustains and merges fewer back-to-back notes.

Without per-clip ground truth on MTG-QBH, the only objective signal is the count: 10/10 still produce notes, no clip silently broke. Quality difference is qualitative (visual SVG inspection) — those SVGs are saved to `outputs/mtg_qbh_soft/` and logged as `wandb.Html` in the run.

## Next
- Hand-annotate 5 MTG-QBH clips with recognizable melodies (Twinkle, etc.) to get a quantitative MTG-QBH F1 — this is the spec §B.2 next step. ~10 min/clip in MuseScore.
- Re-run after B6 (HMM) and B8 (medium/hard sweeps) ship as comparison baselines.
