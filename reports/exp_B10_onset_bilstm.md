# exp_B10_onset_bilstm — BiLSTM onset detector trained on Vocadito

## Goal
Replace the voicing-rising-edge onset trigger with a learned per-frame onset probability from a small BiLSTM trained on Vocadito A1. Hypothesis: temporal context across 5-10 frames lifts onset detection precision/recall above the heuristic.

## Procedure
- Architecture: `humscribe.train.onset_bilstm.OnsetBiLSTM`. 2-layer bidirectional LSTM, hidden 64, dropout 0.2. Output head: 2-layer MLP, scalar logit per frame.
- Features per 10ms frame (in_dim=3): normalized PESTO MIDI estimate, voicing confidence, voicing^2 as energy proxy.
- Labels: onset times from Vocadito CSV → 1.0 in a ±1-frame window around each onset.
- Train/val: 30 / 10 clip random split (seed 0); pos-weighted BCE. AdamW lr=1e-3, 80 epochs, batch=4. GPU.
- Decode: per-frame logit → sigmoid → threshold τ → suppress within 5-frame window of a previous trigger. Threshold swept over {0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97}.
- Eval: same `mir_eval` COnP F1 as Vocadito gate.

## Results
| τ | mean_F1 | mean_P | mean_R |
|---|---|---|---|
| 0.40 | 0.427 | 0.304 | 0.752 |
| 0.50 | 0.455 | 0.370 | 0.615 |
| **0.60** | **0.490** | 0.466 | 0.527 |
| 0.70 | 0.485 | 0.520 | 0.464 |
| 0.80 | 0.421 | 0.583 | 0.335 |
| 0.85 | 0.423 | 0.604 | 0.330 |
| 0.90 | 0.420 | 0.617 | 0.323 |
| ≥0.93 | < 0.34 | — | — |

Voicing-segmenter baseline (B2-tuned, same 10 val clips approx): F1 ≈ 0.55.

WandB train: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/l4m7z0ef
Threshold sweep log: `logs/sweep_onset_threshold.log`

## Interpretation
The BiLSTM hits 0.490 F1 on held-out clips — **about 0.06 F1 below** the voicing baseline (0.577 on full set, ~0.55 on the 10-clip val subset). The model learns: high recall at low threshold (the model fires at most segment boundaries — over-triggers), high precision but low recall at high threshold (model is overconfident in the few crisp onsets it sees).

Three reasons it doesn't beat the heuristic:
1. **Too little training data**: 30 Vocadito clips × 30 s = ~15 min of audio. Modern onset detectors are trained on hundreds of hours.
2. **Features are sparse**: PESTO midi + voicing alone misses spectral cues that real onset detectors use (mel-spectrogram, energy derivatives, etc.).
3. **Eval bias**: same 10 clips for held-out — enough variance per-clip to make the comparison noisy. Cross-validation needed for a fair estimate.

The BiLSTM does have a strong precision regime (P=0.62 at τ=0.9) — it could be used as a precision-filter on the voicing baseline's recall-heavy onsets, but expected lift is small.

Decision: don't promote BiLSTM. File as "needs more data + spectral features".

## Next
- Try training on Vocadito + MIR-1K (MIR-1K has per-frame voicing labels but not note-onset labels, so MIR-1K only helps the auxiliary voicing head).
- B10b: add log-mel features (32 bands) — larger feature dim → more discriminative.
- B10c: try a Transformer-based local-window onset detector (e.g. similar to the onset head in basic-pitch's ICASSP 2022 paper).
- Or skip and pursue B11 (per-clip ensemble of voicing/HMM with selector) — tractable without more data.
