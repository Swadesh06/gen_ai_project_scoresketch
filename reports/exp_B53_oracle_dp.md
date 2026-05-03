# exp_B53 — oracle-input DP test on ASAP

## Goal
Diagnose where the loss comes from on Romantic ASAP (mean snap 0.590 with our pipeline).
Replace the noisy ByteDance + beat_this front-end with the GROUND TRUTH MIDI notes and
GT MIDI beats fed directly into the DP+VT stack. The remaining gap is then 100% DP/VT.

## Procedure
- For each piece, load `midi_score.mid`, take notes verbatim as both GT and "predicted".
- Use `pretty_midi.PrettyMIDI.get_beats()` for the beat grid (from MIDI tempo events).
- Run `assign_voices` with `adaptive_pitch_jump`, then `viterbi_quantize_rhythm`.
- Score `snap %` against GT durations (in beats).

## Results

| piece | pj (adaptive) | oracle snap | actual snap (B49) | gap (upstream loss) |
|---|---|---|---|---|
| Bach Fugue BWV 846 | 3 | **0.925** | 0.847 | -7.8pp |
| Beethoven Sonata 21-1 | 12 | **0.982** | 0.811 | -17.1pp |
| Schumann Toccata | 3 | **0.975** | 0.745 | -23.0pp |
| Chopin Berceuse | 3 | **0.742** | 0.469 | -27.3pp |
| Liszt Sonata | 7 | **0.132** | 0.078 | -5.4pp |
| **mean** | — | **0.751** | **0.590** | **-16.1pp** |

## Interpretation

Two distinct failure modes:

### (1) Upstream-bound: Beethoven/Schumann/Chopin
Oracle scores 0.74–0.98 but actual scores 0.47–0.81. The DP+VT itself works fine; the
problem is ByteDance missing notes / mis-detecting onset times, or beat_this estimating
the wrong beat positions on Romantic-style audio. Fix path:
- B54: try score-aligned beats (Madmom RNNDownBeatProcessor or alignment to GT beats from a different transcription pass).
- B55: try YourMT3+ as a more robust piano transcriber for Romantic chordal textures.

### (2) DP-bound: Liszt
Oracle is **0.132** — even with perfect note inputs, our DP+VT only snaps 13% of
durations correctly on Liszt. This rules out upstream causes for Liszt entirely.
The structural issue is one of:
- `adaptive_pj=7` is wrong for Liszt (try pj=24 to allow 2-octave jumps in voice tracking).
- TPB=24 too coarse for Liszt's 32nd-note runs (try TPB=48).
- `allowed_durations` excludes Romantic-specific values (try adding 5/16, 7/16, complex tuplets).

## Cumulative ASAP picture

| metric | value |
|---|---|
| Bach Fugue (5 pieces, B12) | 0.856 |
| Mixed (1 Bach + 4 Romantic, B49 actual) | 0.590 |
| Mixed (1 Bach + 4 Romantic, B53 oracle) | 0.751 |
| Headroom from upstream improvements | ~16pp |
| Headroom from DP fixes (Liszt only) | ~85pp on Liszt alone (0.13 → 1.0 theoretically) |

## Next
- B54: Liszt-only sweep over pj∈{12, 24} × TPB∈{24, 48} × extended allowed_durations.
- B55: Beethoven/Schumann/Chopin pipeline-improvement (try Madmom beat tracking + maybe YourMT3+).
