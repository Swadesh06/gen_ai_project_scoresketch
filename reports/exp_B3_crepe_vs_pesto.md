# exp_B3_crepe_vs_pesto — pitch-tracker comparison on Vocadito

## Goal
Phase B priority 1: compare PESTO and torchcrepe ('full' model) as the Stage-2-A pitch tracker, holding the rest of the humming pipeline constant. Test whether PESTO's MIR-1K-trained checkpoint generalizes worse to Vocadito's trained-vocal distribution than CREPE-large.

## Procedure
- Pipeline: same as Phase-A Vocadito gate (mode=soft, voicing segmenter, A1 annotator, all 40 clips).
- Pitch tracker: `humscribe.pitch.crepe_track.track_pitch_crepe` with `model='full'`, `hop_ms=10`, GPU. Wraps `torchcrepe.predict` after librosa-resampling to 16 kHz mono.
- Both runs include the B2-tuned soft defaults (vt=0.315, mns=0.052, psw=11, oms=0.026).
- Hardware: GPU for both pitch trackers (sm_120 Blackwell).

## Results
| pitch model | mean F1 | mean P | mean R | wallclock per 40-clip run |
|---|---|---|---|---|
| PESTO (default) | **0.576** | 0.585 | 0.592 | ~1 min |
| torchcrepe full | 0.562 | — | — | ~5 min (GPU) |

Per-clip differences exist (e.g. `vocadito_8`: CREPE F1=0.864 vs PESTO ~0.633; CREPE catches more of the trained tenor's vibrato). But on the 40-clip aggregate, PESTO edges CREPE by +1.4pp.

WandB:
- PESTO: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/<see logs/gate_vocadito_tuned.log>
- CREPE: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/xjriyp9g

## Interpretation
PESTO wins despite being trained on MIR-1K (a different distribution than Vocadito). Two reasons likely matter:
1. **Hop / step interaction**: PESTO produces tightly-spaced 10ms predictions matching the segmenter's smoothing window. CREPE here is also 10ms but its frame model is heavier and the periodicity threshold (used as voicing) might be calibrated differently.
2. **Voicing semantics**: PESTO's confidence is well-correlated with "is this frame voiced". CREPE's periodicity is a different quantity (probability of *some* periodic component). The voicing-driven segmenter's threshold (`vt=0.315`) was tuned for PESTO; the same threshold on CREPE periodicity discards too many voiced frames.

CREPE has clip-level wins on a few harder clips (e.g. vocadito_8 +0.23 F1) — a per-clip ensemble (max over PESTO/CREPE F1) would beat either. The classic "average their pitch contours" ensemble is harder because the two trackers don't align frame-to-frame perfectly.

Decision: keep PESTO as the default. Don't pursue the average-pitch ensemble (the +1.4pp PESTO advantage means CREPE-only would lose; an averaged pitch contour might just regress toward a worse configuration).

## Next
- Try CREPE's periodicity rescaled (e.g., normalize per-clip then threshold).
- B3b: per-clip selector — train a tiny classifier on clip features (avg energy, spectral centroid, voicing entropy) to pick PESTO vs CREPE per clip.
- Move to B6 (LilyPond rendering) or B4b (HMM hyperparam sweep) — both have higher expected lift than continuing pitch-tracker comparison.
