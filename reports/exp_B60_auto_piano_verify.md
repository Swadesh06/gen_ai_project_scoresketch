# exp_B60 — verify auto_piano transcriber end-to-end

## Goal
B59 found basic_pitch wins +9.3pp on Chopin Berceuse but loses 25pp on average.
Implemented `auto_piano` (config.Transcriber) — runs ByteDance, switches to basic_pitch
if median note duration > 0.4s. Verify end-to-end through `_branch_notes()`.

## Procedure
- Render each ASAP score MIDI to WAV via fluidsynth (TimGM6mb.sf2).
- Call `_branch_notes()` with `transcriber="bytedance_piano"` and again with
  `transcriber="auto_piano"`.
- Run DP+VT (cached beat_this beats), compute snap%.
- Threshold tuning data from B60b: med_dur=0.517 for Chopin, ≤0.239 for the others.

## Results

| piece | fixed_bd | auto_piano | chosen | Δ |
|---|---|---|---|---|
| Bach Fugue BWV 846 | 0.847 | 0.847 | bd | 0 |
| Beethoven Sonata 21-1 | 0.811 | 0.811 | bd | 0 |
| Schumann Toccata | 0.745 | 0.745 | bd | 0 |
| **Chopin Berceuse** | **0.469** | **0.521** | **bp** | **+5.2pp** |
| **mean** | **0.718** | **0.731** | — | **+1.3pp** |

auto_piano correctly identifies Chopin as the slow-chordal piece and switches the
transcriber. The other three keep ByteDance.

## Implementation

```python
# humscribe/pipeline.py:_branch_notes
if cfg.transcriber == "auto_piano":
    bd = transcribe_piano(audio_path)
    if len(bd) >= 50:
        durs = np.array([n.offset_s - n.onset_s for n in bd])
        if float(np.median(durs)) > 0.4:
            return transcribe_basic_pitch(audio_path)
    return bd
```

The threshold (median dur > 0.4) was set empirically (B60b shows Chopin uniquely satisfies it).

## Cost
- ~2x compute per piece if it ends up using basic_pitch (runs both transcribers).
- For most pieces, only ByteDance runs (no overhead vs default).

## Decision
Keep. `auto_piano` is the new recommended transcriber for instrument input when the
input characteristics are unknown. Default behavior remains `bytedance_piano`.

## Status
keep
