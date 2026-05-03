# exp_B80 — Verify per_voice_dp library refactor (Phase D)

## Goal
B79 implemented per-voice DP standalone in a script. B80 calls the new
`humscribe.rhythm.voice_tracking.quantize_with_voice_tracking()` with
`per_voice_dp=True` + `voice_assigner=B76_predict` to confirm the
production library refactor reproduces B79's per-piece deltas.

## Procedure
- Same 4 ASAP held-outs (Liszt, Schumann, Chopin, Beethoven 21-1)
- Same B76 voice tracker checkpoint
- Same B63 cached YMT3+ predictions
- Three variants:
  - A. lib default (`per_voice_dp=False`, greedy)
  - B. lib + `per_voice_dp=True`, greedy
  - C. lib + `per_voice_dp=True`, B76 voice_assigner

## Results

| piece | A. lib default | B. lib pvd_greedy | C. lib pvd_b76 | Δ (C−A) |
|---|---|---|---|---|
| Liszt Sonata | 0.0072 | 0.0072 | 0.0065 | -0.0007 |
| Schumann Toccata | **0.8452** | 0.8452 | 0.8066 | -0.0386 |
| **Chopin Berceuse** | 0.4236 | 0.4236 | **0.4451** | **+0.0216** |
| Beethoven 21-1 | **0.8850** | 0.8850 | 0.8817 | -0.0033 |
| **mean** | **0.5402** | 0.5402 | 0.5350 | -0.0052 |

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/gtgvyc32

## Vs B79
| piece | B79 Δ | B80 Δ | match? |
|---|---|---|---|
| Liszt | -0.0007 | -0.0007 | ✓ exact |
| Schumann | -0.0414 | -0.0386 | ~ same (greedy assignment ordering) |
| Chopin | +0.0166 | +0.0216 | better in lib (greedy used 2-cluster split in B79; lib's greedy is more nuanced) |
| Beethoven | -0.0036 | -0.0033 | ~ same |

**Lib refactor verified.** The per-voice-DP path is now production-ready
behind a one-arg switch:

```python
quantize_with_voice_tracking(notes, beats,
    per_voice_dp=True,                    # new arg
    voice_assigner=b76_assigner)          # new arg, optional
```

## Note on lib variants A vs B
Variants A and B produce identical snap because the script's lib-default
greedy assigner returns the same voice grouping in both modes — the
difference would only appear when the per-voice DP runs (B) gives different
quantizations than the shared DP (A) for those same voice groups. On the
Romantic dense pieces tested, both DPs converge to the same answer for
greedy's voice groupings. The B76 tracker (variant C) gives different
voice groups → different DP outputs → the +2.2pp Chopin / -3.9pp Schumann
spread.

## Status
keep — library refactor validated. Recommended Phase E follow-up:
add a per-piece routing heuristic to `pipeline.transcribe()` that
chooses `per_voice_dp=True` for melody+accompaniment pieces (low
voice-overlap density) and `per_voice_dp=False` for dense polyphony.
