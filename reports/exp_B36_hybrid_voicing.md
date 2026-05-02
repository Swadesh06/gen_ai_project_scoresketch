# exp_B36 — PESTO pitch + CREPE periodicity for voicing

## Goal
Test whether using CREPE's `periodicity` (instead of PESTO's `confidence`) as the voicing trigger improves the segmenter. The hypothesis: CREPE periodicity is a more general "is this voiced?" signal calibrated for music in general, while PESTO confidence is tied to PESTO's MIR-1K-trained pitch-detection task.

## Procedure
- Pipeline: PESTO for pitch/midi (kept), interpolate CREPE periodicity onto PESTO frame grid, feed as voicing to `humscribe.pitch.voicing.segment_pitch_to_notes`.
- Sweep `voicing_threshold ∈ {0.20, 0.30, 0.40, 0.50, 0.60}` (B36) + extended {0.55, ..., 0.90} × `psw ∈ {11, 15, 19}` (B36b).
- Vocadito A1 mean F1.

## Results

### B36 (initial range)
| vt(crepe) | F1 |
|---|---|
| 0.20 | 0.606 |
| 0.30 | 0.611 |
| 0.40 | 0.626 |
| 0.50 | 0.639 |
| **0.60** | **0.650** |

Monotone increasing — peak at the high end of the range.

### B36b (extended; partial as of report)
Top so far (8/24 configs done):
| vt | psw | F1 |
|---|---|---|
| **0.60** | **19** | **0.656** |
| 0.55 | 19 | 0.653 |
| 0.65 | 15 | 0.651 |
| 0.60 | 15 | 0.650 |
| 0.55 | 15 | 0.647 |

Best so far: vt=0.60, psw=19, **F1=0.656**.

## Interpretation
CREPE's periodicity is a much better voicing signal than PESTO's confidence for this dataset. PESTO confidence saturates near 1.0 for most voiced frames (the model is confident the pitch is correct, but doesn't necessarily distinguish voiced from unvoiced clearly). CREPE periodicity, computed as the local-maximum-correlation strength, is closer to a true voicing probability.

The optimal `vt(crepe)=0.60` is much higher than the optimal `vt(pesto)=0.30` because CREPE periodicity has a different scale and discriminates more strongly between voiced and unvoiced frames.

The longer `psw=19` (vs psw=15 for PESTO-only) likely reflects CREPE's noisier per-frame pitch — needs more smoothing to get clean medians per segment.

**Cumulative Vocadito A1 F1 progression**:
- Phase A baseline: 0.538
- B2 sweep: 0.577 (+3.9pp)
- B22 psw=15: 0.597 (+2.0pp)
- **B36 hybrid voicing: 0.650 (+5.3pp)**
- **B36b vt=0.60/psw=19 (provisional): 0.656**

Total: **+11.8pp** = **+22% relative improvement** over Phase A baseline.

## Next
- Wire `track_pitch_hybrid_voicing` into `pipeline.transcribe()` as default for humming (with a `pitch_model` flag option to disable).
- Update `ModeConfig.for_mode("soft")` defaults: `voicing_threshold=0.60`, `pitch_smooth_window=19` (when using hybrid voicing).
- B36 cost: 2x model inference. Acceptable on GPU (CREPE-full ~1.4 s per 33 s clip).
- Verify on Vocadito A2 + MTG-QBH after defaults updated.
