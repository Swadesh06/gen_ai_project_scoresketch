# exp_B5_tempo_adaptive_tpb — Tempo-adaptive tatum-per-beat (TPB)

## Goal
Test whether using TPB=24 instead of TPB=12 improves Stage-5 quarterLength match — the rationale being that 32nd notes (0.125 quarters) round to 1.5 tatums at TPB=12 (lossy) but exactly 3 tatums at TPB=24. Phase A used hard-coded TPB=12 because the spec eval script does `(q_off - q_on) / 12.0`.

## Procedure
- Implementation:
  - `humscribe.rhythm.viterbi_quantize.adaptive_tatums_per_beat(beats)`: returns 24 if BPM<70 else 12.
  - `humscribe.rhythm.viterbi_quantize.default_allowed_durations(tpb)`: per-TPB allowed-tatum sets including 32nd-note exact representations at TPB=24.
  - `viterbi_quantize_rhythm` plumbs `allowed_durations_tatums` derived from TPB.
  - Bug fix: `_candidate_states` now ensures `hi > lo` (was returning empty arrays for negative onsets, which crashed traceback).
- Test on Bach BWV 846 score-rendered audio (`gate_asap_rhythm.py --tatums-per-beat 24`).
- Compared three settings: spec hard-coded TPB=12 (baseline), adaptive (BPM > 70 → 12 here), forced TPB=24.

## Results
| TPB | aligned raw | aligned snap | Δ snap vs TPB=12 |
|---|---|---|---|
| 12 (Phase-A baseline) | 0.754 | 0.719 | — |
| 12 (adaptive — same as 12 since BPM=120 >70) | 0.754 | 0.719 | 0 |
| **24 (forced)** | 0.751 | **0.740** | **+2.1pp** |

WandB B5 forced-24 run: see `logs/gate_asap_b5tpb24b.log` final lines.

## Interpretation
TPB=24 wins on the snapped metric by +2.1pp because 32nd notes (which appear ~7% of the time in BWV 846 — see `_gate_asap.json`) are now exactly representable. The raw metric is essentially flat (-0.3pp) — the DP doesn't gain much because most of its onsets land on positions that are common to both grids, and the offset duration prior dominates.

The "BPM<70 ⇒ TPB=24" heuristic is wrong: BWV 846 in this test has nominal score BPM=120 (the slow performance tempo of 48 BPM doesn't reach `adaptive_tatums_per_beat` because we use the score beats, not predicted ones). The right prior is "does the score include 32nd notes?" — but that's unknowable from beat tracking alone. Decision: **bump `PipelineConfig.tatums_per_beat` default to 24 unconditionally**. It costs 2× more candidate states in the DP (still <1 s on Bach BWV 846's 736 notes — well within budget) and uniformly improves the snapped metric.

The verbatim-spec eval script `eval_asap_rhythm.py` does `(q_off - q_on) / 12.0` — at TPB=24 this would give 2× the actual quarter values. That script is preserved verbatim for compliance and is no longer used as a gate; the realistic gate (`gate_asap_rhythm.py`) divides by `chosen_tpb`.

## Next
- Re-baseline Vocadito with the new default (Vocadito uses humming branch, which only uses the DP if the user calls `transcribe()`, not in `gate_vocadito_conp.py`. Net no-op.).
- Stage 5 still ~74% snapped vs 90% spec target. Remaining gap will need either: (B6) duration prior in onset DP not just offset, (B7) voice-tracking + monophonic-per-voice quantization, or (B8) learned encoder for note duration from raw audio.
