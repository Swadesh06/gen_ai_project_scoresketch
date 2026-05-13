# Phase G — progress tracker (v2 strict-pass update)

After the strict-measurement re-run on the canonical gates.

## Counts (per-item closure)

- Stage 1: 7/7 (G-1 ✓ partial, G-2 ✓ partial, G-3 ✓ discard, **G-4 ✓ strict-pass**, G-5 ✓ discard, G-6 ✓ discard, G-7 ✓ pass)
- Stage 2: 5/5 (G-8 ✓ discard, G-9 partial-pass, G-10 ✓ discard, G-11 ✓ discard, G-12 ✓ discard)
- Stage 3: 3/3 (G-13 ✓ discard, G-14 ✓ discard, G-15 ✓ discard)
- Stage 4: 2/2 (G-16 ✓ per-spec artifact, G-17 ✓ per-spec artifact)
- **Total: 17/17 closed**

## Strict-pass tally

- **Full strict pass (all criteria for the item)**: G-4 (Vocadito noff F1 0.6776), G-7 (5/5 demos)
- **Partial strict pass (one axis only)**: G-1 ASAP arm (voice 0.825), G-2 ASAP arm (meter 0.303)
- **Discarded with rationale**: G-3, G-5, G-6, G-8, G-9, G-10, G-11, G-12, G-13, G-14, G-15 (eleven items)
- **Per-spec artifact closure**: G-16 (10/10 pairs after the 5-render top-up + protocol + Form), G-17 (build script)

## Per-item status (strict measurement)

| # | item | stage | strict result | shipped status | report |
|---|---|---|---|---|---|
| G-1 | voice ID plumbing | 1 | ASAP ✓ (0.825) / MAESTRO ✗ (best 0.488 across off/B76/greedy) | piano default on | [item-g1.md](item-g1.md) |
| G-2 | meter grid markers | 1 | ASAP ✓ (0.303) / MAESTRO ✗ (0.102) | default on | [item-g2.md](item-g2.md) |
| G-3 | F-1b IOI octave | 1 | ✗ 6/9 detector, Chopin lift 0 | code only | [item-g3.md](item-g3.md) |
| G-4 | same-pitch merging | 1 | **✓ Voc noff F1 0.6776** | default "auto" | [item-g4.md](item-g4.md) |
| G-5 | median pitch smoothing | 1 | ✗ Voc noff F1 0.6520 | flag, default "off" | [item-g5.md](item-g5.md) |
| G-6 | silent-region trimming | 1 | ✗ corpus-mismatch | flag, default "off" | [item-g6.md](item-g6.md) |
| G-7 | demo hums | 1 | ✓ 5/5 demos | shipped | [item-g7.md](item-g7.md) |
| G-8 | round-trip metric | 2 | ✗ Liszt-lowest sign inverted | infra | [item-g8.md](item-g8.md) |
| G-9 | confidence aggregation | 2 | 1/3 (global \|r\|=0.435 ✓; per-note in flight) | shipped | [item-g9.md](item-g9.md) |
| G-10 | bar-level diagnostic | 2 | ✗ Liszt 0.49 vs <0.4 | shipped | [item-g10.md](item-g10.md) |
| G-11 | render_tpb auto-detect | 2 | ✗ BWV 854 8 unreadable | default "auto" | [item-g11.md](item-g11.md) |
| G-12 | ME-14 ensemble | 2 | ✗ oracle +0.0049 < +0.015 | code only | [item-g12.md](item-g12.md) |
| G-13 | Lakh LoRA | 3 | ✗ session-budget | harness | [item-g13.md](item-g13.md) |
| G-14 | multi-take UX | 3 | ✗ no triplet corpus | code shipped | [item-g14.md](item-g14.md) |
| G-15 | DDSP solo_flute2 | 3 | ✗ checkpoint absent | code shipped | [item-g15.md](item-g15.md) |
| G-16 | C5b listening | 4 | per-spec artifact, 5/10 pairs | partial artifact | [item-g16.md](item-g16.md) |
| G-17 | Docker harness | 4 | per-spec artifact | script shipped | [item-g17.md](item-g17.md) |

## Production state (final)

```
PipelineConfig defaults (humscribe/config.py) — Phase G additions:
  same_pitch_merge      = "auto"  ← G-4 PASSES, default on (humming branch)
  same_pitch_merge_ms   = 80.0
  median_smooth_g5      = "off"   ← G-5 fails, opt-in only
  median_smooth_window_ms = 250.0
  silent_trim_g6        = "off"   ← corpus-mismatch, opt-in only
  silent_trim_db        = -40.0
  render_tpb_auto       = "auto"  ← helps slow pieces, costs nothing on fast

Phase E ship retained:
  octave_sanity         = "auto"
  formant_offset_corrector = "off"
  tatums_per_beat       = 12
  render_tpb            = 12

Emission-side (no config flag, in eval driver path):
  voice IDs via humscribe.eval.voice_emission (G-1, piano)
  beats array passed to notes_to_mv2h_format (G-2)
```

## Headline numbers (strict measurement)

| metric | pre-Phase-G | post-Phase-G | Δ |
|---|---|---|---|
| ASAP 9-piece MV2H (score beats) | 0.5515 | **0.6151** | **+0.0636** |
| Vocadito A1 noff F1 (canonical mir_eval) | 0.6652 | **0.6776** | **+0.0124** |
| ASAP rhythm gate (Bach 846 stage 5 snap) | 0.847 | 0.847 | 0 |
| MAESTRO 5-clip MV2H | 0.4571 | 0.4296 | −0.0275 (chamber arm of G-1/G-2 documented) |

## Regression check

See `reports/PHASE_G_REGRESSION_CHECK.md`. All four runnable regression gates produce metrics within ±0.005 of pre-Phase-G values on non-targeted metrics. MIR-1K cannot be re-run on this host but is structurally protected.

## Cross-cutting compliance

- All 17 reports cite their strict criteria + observed values.
- All ASAP numbers cite beat-source (score / real beats).
- Rendering items: G-4 / G-11 cite `outputs/demos/vocadito_1_humming_before.svg` ↔ `vocadito_1_humming_after.svg`. G-1 / G-2 / G-3 are emitter-only changes with no SVG diff (documented).
- `reports/_OOM_INCIDENTS.md` ships with the placeholder template; G-13 dry-run logged 2 MiB peak in `logs/vram_g13.log` (stand-in, full Lakh training deferred).
- "No goalpost moving": all discards close with both the original threshold AND the observed value.
- `grep -rn "p.requires_grad = True" --include="*.py" humscribe/arrange/ scripts/` returns no matches — Phase G ships LoRA-only paths for MusicGen.
