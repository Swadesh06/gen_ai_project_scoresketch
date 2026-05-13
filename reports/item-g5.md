# item-g5 — median pitch smoothing (Mauch 2014 pYIN)

## Goal
task_description_v4.md item G-5. 250 ms voiced-only moving median on the pitch trace before segmentation. Strict pass: Vocadito A1 noff F1 ≥ 0.67, no instrument regression.

## Procedure
- `humscribe/post_process.py:median_smooth_pitch(times, hz, voicing, window_ms=250.0)`. Centred sliding median across voiced frames only; unvoiced frames keep their original hz.
- Strict gate: `scripts/gate_vocadito_conp_phase_g.py --apply g5 --annotator A1` on the full 40-clip A1 corpus.

## Results — strict noff F1

| state | mean F1 | mean P | mean R | median F1 | n |
|---|---|---|---|---|---|
| baseline (all post off) | 0.6652 | 0.6790 | 0.6615 | 0.6571 | 40 |
| **G-5 alone (--apply g5)** | **0.6520** | 0.6747 | 0.6429 | 0.6516 | 40 |

**G-5 alone Δ = −0.0132 mean F1; does NOT clear the strict ≥ 0.67 threshold.**

The 250 ms voiced-only window pushes farther than the segmenter's existing 190 ms median (`pitch_smooth_window = 19` at 10 ms hop). Both P and R drop slightly:
- P −0.0043: the wider smoothing shifts the per-note pitch median by enough to fail the 50-cent strict pitch tolerance on some matched-onset notes.
- R −0.0186: same mechanism — predicted notes that would have matched a GT note are pushed outside the pitch tolerance.

### Instrument regression
G-5 is gated by `cfg.is_humming()`. ASAP/MAESTRO eval paths bypass it; structurally no instrument regression possible.

## Interpretation
Mauch 2014's published 250 ms window was tuned against a different segmenter family (pYIN's HMM). HumScribe's voiced-segment + median-pitch-change segmenter already has a 190 ms internal median; widening further on the same frame stream over-smooths real pitch transitions, particularly the rapid pitch changes at the start of melismatic Vocadito clips.

Pre-Phase-G the segmenter's 190 ms window is already at the sweet spot for this segmenter family. G-5 doesn't transplant cleanly from one segmenter to another.

## Pass / discard
- **Vocadito A1 noff F1 ≥ 0.67**: original 0.67, observed 0.6520 → **discarded-with-failure-mode-rationale**.
- **No instrument regression**: structurally protected by `cfg.is_humming()` gate → passes mechanism-evidence.

**Net G-5 status: DISCARDED. Default flipped to "off" in `humscribe/config.py`. The flag is preserved for opt-in.**

## Next
Phase H: window-size sweep at 200/220/240/260/280/300 ms to find the optimal for this segmenter family, or replace the segmenter's internal median with G-5 (rather than stacking).
