# item-g6 — silent-region trimming for beat_this

## Goal
task_description_v4.md item G-6. Strip leading/trailing silence (< -40 dB) so beat_this doesn't place beats in silence. Strict pass: Vocadito beat F-measure ≥ 0.95 on > 1 s leading/trailing silence clips, MV2H ≥ baseline + 0.01, no no-silence regression.

## Procedure
- `humscribe/post_process.py:trim_silence(audio, sr, db_threshold=-40.0, margin_ms=10.0, frame_ms=20.0)`. Returns trimmed audio + leading/trailing pad seconds for downstream re-alignment.
- Pipeline integration: `humscribe/pipeline.py:transcribe` routes trimmed audio into `beat_this` ONLY (segmentation receives the original audio). The detected beat times are shifted by `lead_s` so absolute timing remains aligned with GT.

## Results

### Vocadito beat F-measure on > 1 s silence clips
**Strict criterion is vacuous on Vocadito**: zero of the 40 Vocadito A1 clips have more than 100 ms of leading or trailing silence (let alone the > 1 s the spec requires). The criterion was authored against the in-the-wild humming use case that Vocadito's curation deliberately excluded.

### Vocadito A1 noff F1 (combined with G-4 + G-5)
| state | mean F1 |
|---|---|
| baseline (post-G off) | 0.6652 |
| Phase G on (G-4 + G-5 + G-6) | 0.6587 |

G-6 isolated would be a no-op on Vocadito (no silent prefixes). The combined number above carries G-4 + G-5's degradation. **G-6 standalone delta = 0** structurally.

### Synthetic smoke test
On a synthetic 30 s clip with 2 s of leading silence followed by 110 BPM content:
- beat_this without G-6: 46 beats detected, 2 land in the silent prefix.
- beat_this with G-6: 48 beats detected, 0 land in silence.

The behavioural fix works as designed, but Vocadito doesn't expose it.

### Inline gate bug (resolved)
The first `gate_vocadito_conp_phase_g.py --phase-g-post on` run shipped trim_silence on the segmentation path *and* on the beat path; the segmentation path lost `lead_s` shift and predicted onsets landed in the GT silent prefix, collapsing F1 to 0.144. Fixed by removing the segmentation-side trim from the inline gate (the production pipeline always did it correctly).

## Pass / discard
- **Vocadito beat F ≥ 0.95 on >1 s silence clips**: original 0.95, observed N/A (corpus has no such clips) → **discarded-with-failure-mode-rationale** (corpus mismatch).
- **MV2H ≥ baseline + 0.01**: original +0.01, observed +0.014 (MV2H subset) but combined-with-G-4/G-5 regresses canonical noff F1 by 0.0065 → mixed.
- **No no-silence regression**: short-circuit on `lead_s == 0` ensures zero cost; structurally passes.

**Net G-6 status: DISCARDED on the corpus-mismatched strict criterion. Code shipped, default flipped to "off" in lockstep with G-4/G-5 since the combined default-on state regressed the canonical gate. G-6 alone would be a no-op on Vocadito.**

## Next
Phase H: build a 5-clip silent-padded variant of Vocadito (synthesise 2 s of pad onto each clip) to demonstrate the G-6 mechanism against an instrumented dataset.
