# item-8 — MAESTRO chamber demo regenerated with rendering polish

## Goal

Item 8 from `task_descriptions/task_description_v3.md`. `results_v2_evaluation.md`
flagged that the previous Phase B+2 rendering-polish wave updated three of four
demo SVGs but left `demos/maestro_chamber3_30s.svg` showing the pre-polish output
(fractional tempo, no key signature, dense 24-lets + 48-lets). Fix it with the
existing `run_diverse_demos.py` script now that the polish (integer BPM,
KrumhanslSchmuckler key, render TPB=12, tuplet-denominator cap) is wired in.

## Procedure

- Activated humscribe env, sourced .env, `python scripts/run_diverse_demos.py`.
- Co-scheduled with item 1 (MV2H module work) on CPU. CPU bound, ~30 s wall.
- Hardware: 16 GB RTX 2000 Ada (not 32 GB Blackwell as CLAUDE.md cites), 48 cores.
- Bug fix made along the way: `humscribe/instrument/yourmt3plus.py` resolved
  the audio path to absolute before entering `_ymt3_cwd()`. The previous code
  passed the relative path verbatim and the chdir broke audio opening on every
  run from outside the project root.

## Results

| metric | pre-polish (`demos_pre_ymt3/maestro_chamber3_30s.svg`) | now (`demos/maestro_chamber3_30s.svg`) |
|---|---|---|
| tempo display | `♩ = 73.17073170731705` | `♩ = 146` (integer) |
| key signature | none | D major (2 sharps) |
| 24-lets | 9 | 2 |
| 48-lets | 3 | 0 |
| 12-lets | 22 | 12 |
| 6-lets | (n/a) | 9 |
| triplets | (n/a) | 61 |

Files:
- `outputs/demos/maestro_chamber3_30s.musicxml` (95 KB)
- `outputs/demos/maestro_chamber3_30s.svg` (248 KB)
- `outputs/demos_pre_ymt3/maestro_chamber3_30s.svg` retained for diff.

## Interpretation

Item 8's primary goal (no more pre-polish output for the demo) is met. The
remaining 2 24-lets come from music21's `makeNotation()` choosing an awkward
24/13 representation for ql values that *are* multiples of 1/12 (e.g. 13/12)
when the surrounding context invites a stretched-tuplet reading rather than a
tied combination of a quarter + 1/12. This is a downstream music21 quirk, not
a metric or tuplet-cap issue — the upstream tatum grid is integer at rtpb=12.

Acceptable as a result: down from 12 unreadable-tuplet glyphs (9×24 + 3×48) to
2. A follow-up could force music21 to prefer ties over fractional-numerator
tuplets via `Stream.makeNotation(inPlace=False)` flags — saved for Phase F.

## Next

Item 8 closed. Continue item 1 (MV2H end-to-end metric) and queue item 7 ME-9
(line-of-fifths spelling) as CPU-only polish work.
