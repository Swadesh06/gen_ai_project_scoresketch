# Phase G — progress tracker

Final state at session close.

## Counts

- Stage 1: 7/7 (G-1 ✓, G-2 ✓, G-3 ✓, G-4 ✓, G-5 ✓, G-6 ✓, G-7 ✓)
- Stage 2: 5/5 (G-8 ✓, G-9 ✓, G-10 ✓, G-11 ✓, G-12 ✓)
- Stage 3: 3/3 (G-13 ✓, G-14 ✓, G-15 ✓)
- Stage 4: 2/2 (G-16 ✓, G-17 ✓)
- **Total: 17/17**

"Closed" = report + JSON shipped with a pass/discard/deferred decision and metric evidence.
"Shipped" in the table below = code in production paths with explicit defaults.

## Per-item status

| # | item | stage | status | headline | mv2h_mean Δ | shipped | report |
|---|---|---|---|---|---|---|---|
| G-1 | voice ID plumbing | 1 | shipped (piano), MAESTRO discard | ASAP voice 0.704 → 0.825 | +0.024 (G-1 only on ASAP) | yes | [item-g1.md](item-g1.md) |
| G-2 | meter grid markers | 1 | shipped | ASAP meter 0.103 → 0.303 | +0.040 (G-2 only on ASAP) | yes | [item-g2.md](item-g2.md) |
| G-3 | F-1b IOI octave detector | 1 | discarded | Chopin lift 0.000 (target +0.04) | 0 | code only | [item-g3.md](item-g3.md) |
| G-4 | same-pitch gap merging | 1 | shipped | Voc value 0.800 → 0.857 | +0.014 (10-clip combined) | yes | [item-g4.md](item-g4.md) |
| G-5 | median pitch smoothing | 1 | shipped | Voc mp 0.754 → 0.772 | +0.014 (10-clip combined) | yes | [item-g5.md](item-g5.md) |
| G-6 | silent-region trimming | 1 | shipped | synthetic beat-in-silence fix | 0 (none of 10 clips have leading silence) | yes | [item-g6.md](item-g6.md) |
| G-7 | demo hums | 1 | passed | 5 demos ready, 5/5 end-to-end | n/a | yes | [item-g7.md](item-g7.md) |
| G-8 | round-trip metric | 2 | discarded (sign inverted) | \|r\|=0.642 but Liszt-lowest | n/a | infra only | [item-g8.md](item-g8.md) |
| G-9 | confidence-aware | 2 | partial (1 of 3) | \|r\|(med, mv2h)=0.435 | n/a | yes | [item-g9.md](item-g9.md) |
| G-10 | bar-level diagnostic | 2 | partial (correlation ✓) | \|r\|=0.440 | n/a | yes | [item-g10.md](item-g10.md) |
| G-11 | render_tpb auto-detect | 2 | passed | 3 tuplets ≤ 5 strict | n/a (render only) | yes | [item-g11.md](item-g11.md) |
| G-12 | ME-14 ensemble | 2 | discarded | oracle lift +0.0049 (target +0.015) | +0.005 (oracle ceiling) | code only | [item-g12.md](item-g12.md) |
| G-13 | Lakh LoRA | 3 | deferred (harness shipped) | OOM protocol + cache absent | n/a | harness | [item-g13.md](item-g13.md) |
| G-14 | multi-take UX | 3 | code shipped, eval deferred | algorithm + UI shipped | n/a | yes | [item-g14.md](item-g14.md) |
| G-15 | DDSP solo_flute2 | 3 | code shipped, ckpt absent | three fixes wired | n/a | code only | [item-g15.md](item-g15.md) |
| G-16 | C5b listening artifact | 4 | human-eval deferred | 5/10 pairs + protocol + Form | n/a | artifact | [item-g16.md](item-g16.md) |
| G-17 | Docker harness | 4 | user-run deferred | verify.sh shipped | n/a | script | [item-g17.md](item-g17.md) |

## Baselines and headline lifts (session close)

ASAP 9-piece MV2H (real beats from beat_this on cached audio, eval_seconds=30, non_aligned MV2H):

| state | mv2h | multi_pitch | voice | meter | value | harmony |
|---|---|---|---|---|---|---|
| pre-Phase-G baseline | 0.5515 | 0.962 | 0.704 | 0.103 | 0.989 | 0.000 |
| post G-1 + G-2 (default-on) | **0.6151** | 0.962 | **0.824** | **0.303** | 0.985 | 0.000 |
| Δ | **+0.0636** | 0 | **+0.120** | **+0.200** | -0.004 | 0 |

MAESTRO 5-clip (pipeline_full, bytedance_piano, real beats):

| state | mv2h | mp | voice | meter | value |
|---|---|---|---|---|---|
| pre-Phase-G baseline | 0.4571 | 0.892 | 0.488 | 0.085 | 0.820 |
| post G-1 + G-2 (default-on) | 0.4296 | 0.892 | 0.348 | 0.102 | 0.807 |
| Δ | -0.028 | 0 | -0.140 | +0.017 | -0.013 |

MAESTRO regression is documented in item-g1.md / item-g2.md as a chamber-vs-piano discard with rationale (B76 trained on piano hands; MAESTRO has multi-instrument GT).

Vocadito 10-clip subset (pipeline_full, pesto_crepevoicing, humming branch):

| state | mv2h | mp | voice | meter | value |
|---|---|---|---|---|---|
| pre-Phase-G baseline | 0.5162 | 0.754 | 1.000 | 0.027 | 0.800 |
| post G-1/G-2/G-4/G-5/G-6 | 0.5299 | 0.772 | 1.000 | 0.021 | **0.857** |
| Δ | +0.014 | +0.018 | 0 | -0.006 | **+0.057** |

The value sub-score lift on humming is the clearest read of G-4 (same-pitch merging consolidates fragmented notes into longer ones, improving duration match).

## Cross-cutting compliance

- All 17 reports cite all 5 MV2H sub-scores plus mv2h_mean where they're measured.
- All ASAP numbers cite the beat source (score beats vs real beats).
- Rendering items G-1/G-2 confirmed no SVG diff (emitter-only changes). G-11 cites `outputs/demos/vocadito_1_humming_before.svg` / `_after.svg`.
- `reports/_OOM_INCIDENTS.md` ships with the placeholder template; G-13 dry-run logged 2.4 GB peak (well below the 14 GB threshold), no OOM events recorded.
- "No goalpost moving": all discards close with both the original threshold AND the observed value.
- `grep -rn "p.requires_grad = True" --include="*.py"` in humscribe/arrange/ + scripts/ returns no matches — Phase G ships LoRA-only paths for MusicGen.

## Phase H residual gaps

- Chopin Berceuse 3× tempo error: needs a 3-tier or learned beat post-corrector.
- MAESTRO chamber voice/meter scores: needs a chamber-specific voice tracker.
- Harmony sub-axis at 0.000: needs a chord-recognition module.
- Vocadito noff F1 canonical measurement with G-4/5/6 default-on (deferred this session).
- G-13 Lakh corpus prep + training (~4-5 hours).
- G-15 solo_flute2 checkpoint download + eval.
- G-16 5 additional listening pairs + actual rater study.
- G-8 round-trip metric redesign (chroma DTW + density normalisation).
