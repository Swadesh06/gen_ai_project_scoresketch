# item-g3 — F-1b second-signal octave detector (IOI signal)

## Goal
task_description_v4.md item G-3. Add absolute-IOI signal to `humscribe/beat/octave_sanity.py`: if `pred_bpm >= 100` and `median_note_ioi >= 0.4 s`, recommend `halve`. The intent is to catch Chopin Berceuse, where the original F-1 `fast_tempo_slow_note` signal misses because `notes_per_beat` is normal in both the wrong and right tempo octaves. Strict pass: 9/9 detector correct on ASAP, Chopin Berceuse MV2H lift ≥ +0.04, no false fires.

## Procedure
- Added `ioi_signal_halve = (pred_bpm >= 100.0 and median_note_ioi >= 0.4)` as an independent third condition next to the existing `fast_tempo_slow_note` and `notes_per_beat < 0.4` halve gates.
- Diagnostic dict extended with `pred_bpm`, `ioi_signal_halve`, and `fast_tempo_slow_note` so downstream eval can pick which signal fired.
- Probe (`source humscribe; python -c "..."`) ran the updated detector on all 9 ASAP pieces using `track_beats_beat_this(target_bpm=110.0)` and the cached YourMT3+ note traces.

## Results

### Detector behavior on 9 ASAP pieces (real beats from beat_this, score-beats reference)

| piece | bpm (pred) | median IOI (s) | npb | F-1 recommend | G-3 IOI signal | G-3 recommend |
|---|---|---|---|---|---|---|
| Bach__Fugue__bwv_846 | 122.4 | 0.13 | 3.77 | keep | False | keep |
| Bach__Fugue__bwv_848 | 120.0 | 0.13 | 3.85 | keep | False | keep |
| Bach__Fugue__bwv_854 | 120.0 | 0.13 | 3.85 | keep | False | keep |
| Bach__Fugue__bwv_856 | 81.1 | 0.12 | 6.17 | double | False | double |
| Bach__Fugue__bwv_857 | 120.0 | 0.13 | 3.85 | keep | False | keep |
| Beethoven__Piano_Sonatas__21-1 | 150.0 | 0.10 | 4.00 | keep | False | keep |
| Schumann__Toccata | 125.0 | 0.12 | 4.00 | keep | False | keep |
| Chopin__Berceuse_op_57 | 120.0 | 0.16 | 3.13 | keep | **False** | **keep** |
| Liszt__Sonata | 115.4 | 0.16 | 3.25 | keep | False | keep |

### Chopin Berceuse MV2H (target piece for G-3)

| state | mv2h | meter | voice | mp | value |
|---|---|---|---|---|---|
| no octave_sanity | 0.5261 | 0.000 | 0.658 | 0.977 | 0.996 |
| F-1 only (no G-3 IOI signal) | 0.5312 | 0.025 | 0.658 | 0.977 | 0.996 |
| **G-3 IOI signal added (this commit)** | **0.5312** | **0.025** | **0.658** | **0.977** | **0.996** |
| G-3 lift on Chopin | **0.0000** | 0 | 0 | 0 | 0 |

### Why the IOI signal doesn't fire on Chopin
The task description estimated `median IOI ≥ 0.4 s` would catch Chopin. On the YourMT3+ transcription, Chopin's median note IOI is **0.16 s** — well below 0.4. Chopin Berceuse contains many ornamental 16th-notes inside the slow 8th-note pulse; the median picks up the ornament tempo rather than the structural beat. Lowering the threshold to 0.15 s makes it fire on Liszt (median IOI 0.16 s) too, which is a true-positive on Liszt that would corrupt that piece's beats and regress its MV2H.

### Structural diagnosis (carry-forward from prior F-1 analysis)
Chopin Berceuse is in 6/8 at ~40 BPM (dotted quarter beat); `beat_this(target_bpm=110)` picks up the 8th-note pulse at 120 BPM = **3×** the score beat rate. `halve` (60 BPM) is still 1.5× off and `double` (240 BPM) is 6× off; neither halve nor double can reach the correct octave. The structural fix is a 3-tier `third` operation (which F-1 / G-3 don't include) or a learned beat post-corrector (Phase H idea F-1 in `reports/PHASE_F_IDEAS.md`).

## Interpretation
- The G-3 IOI signal is a CORRECT INCREMENTAL DEFENSE that simply doesn't trigger on any of the 9-piece ASAP test set under the YourMT3+ note distribution. It is preserved in code because it costs nothing and may catch a humming-side case (where median IOI is structurally larger).
- The strict criterion `Chopin Berceuse MV2H lift ≥ +0.04` is unmet. The task description's expected mechanism (IOI ≥ 0.4 s as a Chopin marker) is wrong on the actual data — Chopin's transcribed IOI is 0.16 s.
- The strict criterion `9/9 detector correct on ASAP` is also unmet: current F-1 + G-3 is 6/9 by the GT labels chosen from prior F-1 narrative (`halve` for BWV 846 / BWV 856 / Chopin), or 7/9 if BWV 856's `double` recommendation is accepted as correct (which was the prior F-1 behavior). Either reading misses Chopin.

## Rendered output diff
G-3 does not fire on any of the 4 demo SVGs (Bach BWV 854, MAESTRO chamber, MTG-QBH q1, Vocadito 1). The rendered SVGs are bit-identical to pre-G-3. No before/after pair is needed.

## Pass / discard
- **9/9 detector correct on ASAP**: original 9/9, observed 6/9 → **discarded-with-failure-mode-rationale**.
- **Chopin Berceuse MV2H lift ≥ +0.04**: original +0.04, observed +0.0000 → **discarded-with-failure-mode-rationale** (Chopin needs a 3-tier correction; halve/double can't reach a 3× error).
- **No false fires**: G-3 fires on 0/9 pieces → no false fires (vacuous pass).

**Net G-3 status: DISCARDED. The IOI signal is shipped in `humscribe/beat/octave_sanity.py` as a defensive defense (no false fires, may help humming downstream) but is not a Chopin Berceuse fix. The correct path to closing Chopin is a learned beat post-corrector with a 3-tier capability — Phase H F-1.**

## Next
G-4 (same-pitch gap merging) — wired into the humming branch; expected lift is on Vocadito offset-strict F1 rather than the 9-piece ASAP MV2H.
