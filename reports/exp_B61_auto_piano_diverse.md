# exp_B61 — auto_piano on Debussy + Brahms (validation)

## Goal
B60's auto_piano (median dur > 0.4 → switch to basic_pitch) won +5.2pp on Chopin
Berceuse. Validate that the heuristic generalizes to other slow Romantic/Impressionist
piano pieces: Debussy "Reflets dans l'Eau" (impressionist) and Brahms op 118-2 (intermezzo).

## Procedure
- Render score MIDI → fluidsynth WAV.
- Run `_branch_notes()` with `bytedance_piano` and `auto_piano` separately.
- Live `track_beats_beat_this(wav)` (vs cached in B60).
- DP+VT, snap%.

## Results

| piece | fixed_bd | auto_piano | chosen | med_dur |
|---|---|---|---|---|
| Bach Fugue BWV 846 | 0.025 | 0.025 | bd | 0.239 |
| Debussy Reflets | 0.196 | 0.172 | **bp** | 0.484 |
| Brahms op 118-2 | 0.488 | 0.343 | **bp** | 0.786 |

**Both bp-routed pieces LOST** (Debussy -2.4pp, Brahms -14.5pp). The auto switch is wrong
for them.

(Note: Bach 0.025 here vs 0.847 in B58 — likely a beat-grid mismatch since B61 uses live
`track_beats_beat_this` while B58 used cached beats; investigate separately.)

## Interpretation

The median-dur > 0.4 threshold over-fires. Both Debussy (med_dur=0.484) and Brahms
(med_dur=0.786) trip the threshold but bp does not help them.

**The Chopin win was specific** to its texture (rocking accompaniment + simple melody,
sparse polyphony) rather than a general property of slow chordal pieces. n=4 in B59 was
too small to extract a reliable rule.

## Decision: revert auto_piano to no-op

Keep the `auto_piano` enum value for API stability but make it a synonym for
`bytedance_piano`. Users who want basic_pitch-on-Chopin should opt in explicitly.

A reliable selector probably needs a small classifier on bd output features
(pitch range distribution, chord density, harmonic spread) — a follow-up project.

## Status
revert (auto_piano = bd until a better selector is built)

## Updated production state
- Default ASAP transcriber: bytedance_piano (unchanged)
- ASAP 5-Bach mean snap: 0.856
- ASAP 5-mixed mean snap: 0.590 (unchanged from B49)
- For Chopin Berceuse specifically, manually pass `transcriber="basic_pitch"` for +9.3pp
