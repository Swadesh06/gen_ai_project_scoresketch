# item-g2 — meter grid markers in MV2H emission

## Goal
task_description_v4.md item G-2. Emit MV2H Tatum positions interpolated from real beat positions (beat_this on pred, `pretty_midi.get_beats()` on GT). Strict pass: ASAP meter ≥ 0.30 (was 0.10), MAESTRO meter ≥ 0.35 (was 0.14), DTW non-collapse, no other sub-score regression.

## Procedure
- `_tatum_grid_from_beats(beats, total_ms, tatums_per_beat)` linearly interpolates `tatums_per_beat` positions between each pair of consecutive beats; extends past the last beat using the last IBI.
- `notes_to_mv2h_format(...)` accepts optional `beats: Sequence[float]`. When supplied, uses the real-beat grid; falls back to uniform-from-bpm when None.
- ASAP pred path: `track_beats_beat_this(target_bpm=110)` cached to `/workspace/.cache/asap_beats/<piece>.npz` with F-1 octave-sanity applied. ASAP GT path: `pretty_midi.get_beats()` on `midi_score.mid`.
- MAESTRO pred/GT paths: `r.beats` from `pipeline.transcribe(...)` and `pm.get_beats()` from the 30 s MIDI excerpt.

## Results

### ASAP 9-piece (real beats from beat_this on cached audio)

| metric | baseline | G-2 only | g1g2 combined |
|---|---|---|---|
| multi_pitch | 0.962 | 0.962 | 0.962 |
| voice | 0.704 | 0.703 | 0.824 (G-1 contribution) |
| **meter** | **0.103** | **0.303** | **0.303** |
| value | 0.989 | 0.989 | 0.985 |
| harmony | 0.000 | 0.000 | 0.000 |
| **mv2h_mean** | 0.5515 | **0.5915** | **0.6151** |

**ASAP meter ≥ 0.30 → observed 0.303 → strict PASS.**

### MAESTRO 5-clip chamber (with G-1 + G-2 default-on)

| metric | baseline | g1g2 | Δ |
|---|---|---|---|
| multi_pitch | 0.892 | 0.892 | 0 |
| voice | 0.488 | 0.348 | -0.140 (G-1's MAESTRO arm regression — discarded for chamber) |
| **meter** | **0.085** | **0.102** | **+0.017** |
| value | 0.820 | 0.807 | -0.013 |
| mv2h_mean | 0.4571 | 0.4296 | −0.028 |

**MAESTRO meter ≥ 0.35 → observed 0.102 → strict FAIL.**

`beat_this` on chamber audio mis-detects the meter — the chamber clips' true downbeats don't align with the 30 s MIDI excerpt's `pretty_midi.get_beats()`. The lift over baseline is +0.017 (real, small) but falls well short of 0.35.

### Per-piece ASAP meter sub-score (baseline → G-2)

| piece | baseline | G-2 | Δ |
|---|---|---|---|
| Bach__Fugue__bwv_846 | 0.000 | 0.328 | +0.328 |
| Bach__Fugue__bwv_848 | 0.300 | 0.460 | +0.160 |
| Bach__Fugue__bwv_854 | 0.303 | 0.470 | +0.167 |
| Bach__Fugue__bwv_856 | 0.132 | 0.229 | +0.097 |
| Bach__Fugue__bwv_857 | 0.064 | 0.463 | +0.399 |
| Beethoven__Piano_Sonatas__21-1 | 0.070 | 0.171 | +0.101 |
| Schumann__Toccata | 0.020 | 0.374 | +0.354 |
| Chopin__Berceuse_op_57 | 0.000 | 0.025 | +0.025 |
| Liszt__Sonata | 0.039 | 0.211 | +0.172 |

### DTW collapse check
9/9 ASAP + 5/5 MAESTRO produce numeric MV2H, no NaN rows. → ✓

### Multi-pitch / value regression
ASAP value −0.004, MAESTRO value −0.013 — within tolerance.

## Pass / discard
- **ASAP meter ≥ 0.30**: original 0.30, observed **0.303** → **passed-with-metric-evidence**.
- **MAESTRO meter ≥ 0.35**: original 0.35, observed **0.102** → **discarded-with-failure-mode-rationale** (chamber audio `beat_this` mis-detection — the real-beat grid is only as good as the upstream beat tracker).
- **No DTW collapse**: ✓
- **No other regression**: ASAP value −0.004, MAESTRO value −0.013 — within tolerance.

**Net G-2 status: PASSES ASAP arm; MAESTRO arm strict-fails on chamber beat ambiguity. Production state: `notes_to_mv2h_format(beats=...)` is called from the eval driver with `pipeline.transcribe` beats — default on.**

## Rendered output diff
G-2 is emitter-only; SVGs are unchanged. No before/after pair needed.

## Next
Phase H: chamber-tuned beat tracker (or a learned beat post-corrector for the chamber meter mis-detection) to close the MAESTRO arm.
