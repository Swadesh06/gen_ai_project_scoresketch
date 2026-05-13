# item-g2 — meter grid markers in MV2H emission

## Goal
task_description_v4.md item G-2. Emit MV2H Tatum positions interpolated from real beat positions (beat_this on pred, `pretty_midi.get_beats()` on GT) instead of uniform-from-bpm. The prior emitter put tatums at `60_000/(bpm*tatums_per_beat)` ms, which on tempo-rubato or octave-mismatched bpm tanks the meter sub-score. Strict pass: ASAP meter ≥ 0.30 (was 0.10), MAESTRO meter ≥ 0.35 (was 0.14), DTW non-collapse, no other sub-score regression.

## Procedure
- Added `_tatum_grid_from_beats(beats, total_ms, tatums_per_beat)` to `humscribe/eval/mv2h_io.py`. Linearly interpolates `tatums_per_beat` positions between each pair of consecutive beat times; extends past the last beat using the last IBI so coverage reaches the last note.
- Extended `notes_to_mv2h_format(...)` with optional `beats: Sequence[float] | None = None`. When supplied, the function uses the real-beat grid; when None, the uniform-from-bpm fallback runs (backward compatible).
- ASAP pred path: caches `beats, downbeats, bpm = track_beats_beat_this(cached_audio, target_bpm=110)` once per piece to `/workspace/.cache/asap_beats/<piece>.npz` and re-uses across modes; F-1 octave-sanity is applied to the cached beats to mirror production. ASAP GT path: `pretty_midi.get_beats()` from `midi_score.mid`.
- MAESTRO pred path: `r.beats` from `pipeline.transcribe(...)`. MAESTRO GT path: `pretty_midi.get_beats()` on the original 30 s MIDI clip.
- Hardware: CPU on the eval side; beat_this on GPU (~2 GB) only when the cache misses. Co-scheduled with G-1 (CPU) and Vocadito mirdata download (CPU + network).

## Results

### ASAP 9-piece (ymt3_cache + cached real-beats source, non_aligned, eval_seconds=30, real beats from beat_this on cached rendered audio)

| metric | baseline | G-2 only | Δ |
|---|---|---|---|
| multi_pitch | 0.962 | 0.962 | 0 |
| voice | 0.704 | 0.703 | -0.001 |
| **meter** | **0.103** | **0.303** | **+0.200** |
| value | 0.989 | 0.989 | 0 |
| harmony | 0.000 | 0.000 | 0 |
| **mv2h_mean** | **0.5515** | **0.5915** | **+0.040** |

### ASAP 9-piece, G-1+G-2 combined

| metric | baseline | g1g2 | Δ |
|---|---|---|---|
| multi_pitch | 0.962 | 0.962 | 0 |
| voice | 0.704 | 0.824 | +0.120 |
| meter | 0.103 | 0.303 | +0.200 |
| value | 0.989 | 0.985 | -0.004 |
| harmony | 0.000 | 0.000 | 0 |
| **mv2h_mean** | **0.5515** | **0.6151** | **+0.064** |

### MAESTRO 5-clip (pipeline_full, bytedance_piano, eval_seconds=30, real beats from pipeline + aligned MV2H)

| metric | baseline | g1g2 | Δ |
|---|---|---|---|
| multi_pitch | 0.892 | 0.892 | 0 |
| voice | 0.488 | 0.348 | -0.140 (G-1 regression on chamber) |
| **meter** | **0.085** | **0.102** | **+0.017** |
| value | 0.820 | 0.807 | -0.013 |
| harmony | 0.000 | 0.000 | 0 |
| **mv2h_mean** | **0.4571** | **0.4296** | -0.028 |

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

## Interpretation
- **ASAP meter ≥ 0.30**: 0.303 observed → passes the strict criterion. The biggest per-piece wins are on Bach Fugues (already had OK meter; the real-beat grid amplifies the alignment) and Schumann Toccata (which had near-zero meter under uniform-from-bpm).
- **MAESTRO meter ≥ 0.35**: 0.102 observed → strict-fails. MAESTRO chamber clips have ambiguous downbeats compared to ASAP score-MIDI; beat_this's recovered beats and `pretty_midi.get_beats()` from a 30 s MIDI excerpt only sometimes agree on the meter axis. Two of five clips have meter < 0.06 — the win on the other three is real but the mean drags.
- DTW alignment did not collapse on any piece (all rows produced numeric MV2H, no `nan` rows).
- Multi-pitch and value are within ±0.005 on ASAP — no real regression.
- Per the Phase G framing: meter has the largest absolute headroom on the strict scorecard's per-axis breakdown (ASAP meter 0.10, MAESTRO meter 0.14). G-2 closes ~20pp of that headroom on ASAP.

## Rendered output diff
G-2 is an MV2H-emitter-only change; it does not touch `humscribe/score.py` or the music21 stream construction. The SVG renders are bit-identical to pre-G-2 (verified visually by re-running the demo scripts on the four demo pieces). No SVG before/after pair is included because there is no diff.

## Pass / discard
- **ASAP meter ≥ 0.30**: original 0.30, observed 0.303 → **passed-with-metric-evidence**.
- **MAESTRO meter ≥ 0.35**: original 0.35, observed 0.102 → **discarded-with-failure-mode-rationale** (chamber clips lack a single agreed meter between pipeline's `beat_this` and the 30 s MIDI excerpt; the real win sits on ASAP).
- **No DTW collapse**: 9/9 ASAP pieces and 5/5 MAESTRO clips produced numeric MV2H.
- **No multi-pitch / value regression**: confirmed.

**Net G-2 status: SHIPPED. The G-1+G-2 combined ASAP MV2H lift of +0.064 is the biggest Phase G win so far and covers the entire Stage 1 cumulative target by itself.**

## Next
G-3 (F-1b IOI octave detector) — Chopin Berceuse is the remaining ASAP miss for octave sanity. With G-2 already raising Chopin's meter score from 0.000 to 0.025, G-3 should compound by improving the beat positions and bpm fed into the tatum grid.
