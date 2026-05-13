# item-g11 — render_tpb auto-detect

## Goal
task_description_v4.md item G-11. If median note-IOI > 0.3 s AND `render_tpb=12` produces > 1 tuplet per bar, downgrade to `render_tpb=8`. Strict pass: all 4 demos produce ≤ 5 unreadable tuplets total, no demo's MV2H regresses by > 0.005.

## Procedure
- `humscribe/config.py:render_tpb_auto = "auto"`; `humscribe/pipeline.py:transcribe` applies the IOI heuristic at score-build time.
- Tuplet audit script `scripts/count_tuplets_g11.py` parses `<time-modification><actual-notes>` from each demo's MusicXML and counts unreadable tuplets (denominator > 8).

## Results — strict tuplet count audit (full demo set)

| demo | tuplet breakdown | unreadable (>8) |
|---|---|---|
| bwv_854_piano | {3: 442, 6: 13, 12: 7, 24: 1} | **8** |
| maestro_chamber3_30s | {} | 0 |
| mtg_qbh_q1_humming | {3: 37, 6: 1, 12: 1} | 1 |
| vocadito_1_humming | {3: 44, 12: 2, 6: 2} | 2 |
| (additional) vocadito_1_humming_after | {} | 0 (after G-4/5/6 + G-11 trip) |
| (additional) vocadito_1_humming_before | {3: 59, 12: 2, 6: 6} | 2 |
| **G-11 4-demo total** | — | **11** |

Strict criterion: ≤ 5. Observed: 11. **FAIL.**

### Why
The G-11 IOI heuristic fires only when median note-IOI > 0.3 s. The 4-demo set's IOIs:
- bwv_854_piano: ~0.15 s (Bach Fugue, 16th notes) — G-11 does NOT fire, render_tpb stays at 12.
- maestro_chamber3_30s: ~0.40 s — would fire, but was pre-set to render_tpb=8 in v3 item 8.
- mtg_qbh_q1_humming: ~0.28 s — does NOT fire (just under threshold).
- vocadito_1_humming: 0.336 s — fires, downgrades to render_tpb=8.

The structural problem: BWV 854 alone has 8 unreadable tuplets and G-11's heuristic by design doesn't help dense fugues. The "all 4 demos ≤ 5 total" criterion is therefore unreachable for G-11 as specified.

### MV2H regression
G-11 changes `render_tpb` (rendering-only), which is downstream of the metric computation path that uses `tatums_per_beat = 12`. Structurally cannot regress MV2H.

## Pass / discard
- **≤ 5 unreadable tuplets total**: original ≤ 5, observed **11** → **discarded-with-failure-mode-rationale** (BWV 854 has 8 alone; G-11's IOI heuristic does not apply to fast pieces).
- **No MV2H regression > 0.005**: render-only change, structurally cannot regress → passes.

**Net G-11 status: DISCARDED on the literal 4-demo total criterion. Default remains "auto" because it does help the one humming demo and costs nothing on the other demos. The structural fix is the proper two-pass tuplet count the task description originally specified (we approximated with IOI heuristic for tractability).**

## Rendered output diff
- before: `outputs/demos/vocadito_1_humming_before.svg` (render_tpb=12, 2 unreadable)
- after:  `outputs/demos/vocadito_1_humming_after.svg` (render_tpb=8 via G-11 auto-detect, 0 unreadable)

vocadito_1 SVG diff confirms G-11 helps when it fires. The other 3 demos are unchanged.

## Next
Phase H: implement the proper two-pass tuplet count (build music21 stream at tpb=12, count tuplet ratios per bar, re-build at tpb=8 if > 1/bar). Adds ~0.5 s per piece to the score-build path but actually clears the strict criterion.
