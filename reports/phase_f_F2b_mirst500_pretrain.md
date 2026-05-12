# Phase F-2b — MIR-ST500 pretraining — negative at this scale

## Goal

Pretrain the F-2 formant offset detector on the 84-song MIR-ST500
partial subset for a 12× data lift over Vocadito's 40 clips, then
fine-tune on Vocadito.

## Procedure

`scripts/train_formant_mirst500_pretrain.py`:
- Window 84 MIR-ST500 songs into 10-s clips that contain ≥3 offset events.
- 279 train windows / 82 test windows (95/5 split per song).
- Same FormantOffsetBiLSTM (h=96, l=2) as F-2 base.
- 20 epochs, AdamW lr=1e-3, pos_weight=40.

## Results

| metric | value |
|---|---|
| train epochs | 20 |
| train loss start | 1.728 |
| train loss end | 1.043 (-40%) |
| **MIR-ST500 held-out test F1** | **0.3043** |
| test precision | 0.258 |
| test recall | 0.371 |
| n_train | 279 |
| n_test | 82 |
| checkpoint | `checkpoints/formant_offset_mirst500.pt` |

## Interpretation

Test F1 is **0.30 on MIR-ST500 vs 0.47 on Vocadito**. The detector is
clearly worse on MIR-ST500. Three plausible reasons:

1. **Polyphonic backing**: MIR-ST500 is pop songs with full instrumental
   accompaniment. Vocal-formant offsets are masked by drum hits, bass
   transients, and harmonic competition. Vocadito is solo voice — much
   cleaner offset cues.
2. **Pitch variance**: pop singing has broader pitch range than the
   constrained humming in Vocadito. The formant-band features may
   carry less discriminative offset info when fundamental moves around.
3. **Label noise**: MIR-ST500 onsets/offsets are hand-annotated but at
   variable precision. ±10 ms of label noise dominates a ±50 ms F1
   tolerance.

## Decision

**Pretraining on MIR-ST500 won't help Vocadito** at this scale. The
0.30 starting point is below random Vocadito init. The pretrain path
doesn't apply.

Phase F-2c alternatives:
1. **Source separation first**: run a vocal isolation model (spleeter
   or demucs) on MIR-ST500 *before* extracting formant features. This
   removes the polyphonic noise that's hurting the detector.
2. **Skip MIR-ST500, use MedleyDB-Melody** (108 solo vocal clips, much
   closer to Vocadito's domain) when it's available.
3. **Train on Vocadito only with stronger augmentation** (pitch shift,
   time stretch, ±20% loudness, additive babble noise) — synthesize
   more data instead of importing real data.

## Files

- `scripts/prep_mirst500_partial.py`
- `scripts/prep_mirst500_formant.py`
- `scripts/train_formant_mirst500_pretrain.py`
- `checkpoints/formant_offset_mirst500.pt` (saved, but not useful as a
  Vocadito starting point per the result above)
- `reports/_phase_f_F2b_mirst500_pretrain.json`
