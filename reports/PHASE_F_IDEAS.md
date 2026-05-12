# Phase F — autonomous ideas after items 1–8

This file enumerates Phase-F ideas to pursue once Phase E's eight items
have landed (kept members), been documented (discarded members), or are
stable enough that the agent can context-switch.

The biggest unfixed gaps from `results_v2_evaluation.md`:
- **27pp ASAP score-vs-real-beats gap** (0.774 vs 0.506 on 9-piece mean)
- **22pp Vocadito offset20 gap** (0.439 vs IAA 0.642)
- **Liszt structural ceiling** (real-beats snap = 0.054, oracle = 0.132)

## F-1. Learned beat post-corrector (highest priority)

**Target**: 27pp ASAP gap. **Compute**: GPU (~2 GB train, ~0.5 GB inference). **CPU companion**: Existing item 6 sweep.

The v2 evaluation's correction said do NOT fine-tune `beat_this` on ASAP
— it was already trained on ASAP + 14 other corpora. So the fix isn't
"more data"; it's a **post-processor** that takes `beat_this`'s output +
audio features and shifts beats to match what the score expects.

Architecture:
- Input: `beat_this` beats (sparse times), `beat_this` raw beat-curve
  output (dense, per-frame), local audio features (mel-spec).
- Output: per-beat shift in [-0.2, +0.2] s.
- Train data: ASAP `midi_score.mid` beats (the GT) vs `beat_this`'s
  detected beats on the rendered audio. ~50 pieces in ASAP, ~30k beats.
- Model: 2-layer Transformer encoder, hidden 128, 1.2M params.
- Loss: smooth-L1 on the shift, masked where the GT beat has no match.
- Eval: re-run the full pipeline on the 9-piece ASAP test set; the headline
  is "snap (real beats)" — should move from 0.5055 toward 0.7+ if the
  corrector works.

This is the **single highest-EV** Phase F idea per the gap analysis.

## F-2. Formant-band onset detector

**Target**: Vocadito offset20 gap. **Compute**: small GPU (~1.5 GB). **CPU companion**: item 6 sweep.

The vocadito offset gap is dominated by vibrato-induced voicing dips that
end notes early (per the v1/v2 evals). The fix is to detect onsets/offsets
from the 1.5–3.5 kHz formant band — where vocal-tract noise is weaker
than in the fundamental band, so vibrato doesn't trigger false offsets.

Architecture:
- Input: 64-bin mel-spectrogram restricted to 1500–3500 Hz, 10 ms hop.
- Output: per-frame offset logit.
- Train: 40 Vocadito clips, 5-fold CV. Augmentation: pitch shift ±2 semi,
  time stretch ±10%, additive noise, formant shift ±5%.
- Pretrain: MIR-ST500 (item 2's dataset) if we can fetch it.
- Eval: Vocadito offset20 F1 should rise above 0.55 (gap closes from
  -22pp to -10pp). MV2H should rise by 0.01–0.02.

## F-3. MV2H-driven system-level ensemble (ME-14)

**Target**: meta-improvement; depends on item 6 finishing. **Compute**: CPU-only.

Run the pipeline at N parameter combinations (best 5 of the item 6 sweep
+ 3 hand-picked baselines + 2 outlier corner-case configs). For each
piece, pick the variant whose output has the highest MV2H against the
GT. Aggregate the per-piece picks into a routing table keyed by piece
features (note density, pitch IQR, BPM range).

This is "the strongest theoretical move" from the v3 spec — using the
actual end-to-end objective to choose between candidates.

## F-4. Lakh MIDI LoRA fine-tune (extension of item 5)

**Target**: MusicGen generalisation. **Compute**: GPU (~12 GB). **Solo**.

After C5 finishes (JSB Chorales LoRA), extend the training to the Lakh
MIDI corpus (~170k pop/rock MIDIs). Render melody + arrangement pairs
the same way, but use multiple SoundFonts for stylistic diversity.

Pass: MusicGen-Melody-Large with the Lakh LoRA produces a recognisable
arrangement when given a humming melody — not just memorising the 6
distill pairs from B77 or the 370 chorales from C5.

## F-5. Tempo-curve preservation in DP

**Target**: structural rubato (Liszt, Romantic pieces). **Compute**: CPU-only.

Current DP uses a constant BPM derived from `beat_this`'s median IBI.
For rubato pieces this destroys the per-beat timing variation. Cemgil-
Kappen DP supports a per-beat IBI sequence — feed the raw IBIs into the
DP instead of averaging. Liszt's structural snap-ceiling at 0.132 (per
B53) suggests this could lift Liszt out of structural failure into the
0.2–0.3 range. Won't fix the dominant 27pp gap (that's a beat-position
gap, not a beat-spacing gap) but addresses a different structural
weakness.

## F-6. Score-conditioned LoRA for MusicGen

**Target**: arrangement quality at demo time. **Compute**: GPU (~12 GB). **Solo**.

The current MusicGen-Melody conditioning is melody-audio + text. Adding
MIDI as a third conditioning input (the transcribed MIDI from Stages 1-6)
gives strictly more information. The LoRA learns to weight the MIDI
condition more heavily for harmonic content while keeping melody for
rhythmic content.

## F-7. Pre-baked demo recordings

**Target**: demo robustness. **Compute**: trivial.

Bundle 5-10 ready-to-load humming recordings (Twinkle, Mary had a Little
Lamb, etc.) in `assets/demos/` so the Streamlit demo works without a mic.
Tiny work; significant UX win for demo day.

## F-8. Text-prompt style hints

**Target**: humming UX. **Compute**: trivial (MiniLM embedding 23M params).

Free-text "jazz waltz" / "jig" / "ballad" hints from the user, embedded
via MiniLM, biasing the DP's complexity prior toward expected note
densities. Lightweight, interpretable. Goes into the existing
PipelineConfig as a `style_hint: str | None` field.

## F-9. Video-diff evaluation outputs

**Target**: course paper figures. **Compute**: CPU-only.

Side-by-side scrolling rendered score (predicted) vs ground-truth score,
synced to audio playback, exported as an mp4. Better narrative figure
than any F1 number.

## F-10. Web-based notation editor

**Target**: flywheel for training data. **Compute**: trivial UI; data accrues over time.

Let users correct transcription mistakes in the rendered SVG; each
correction logs an (audio, correct-notation) pair to a flywheel database.
Long-term play. 100 corrected hums beats Vocadito's 40 by 2.5× at no
new annotation cost.

## Priority order

For the remainder of this session, after items 1-8 are stable:
1. **F-1** (learned beat post-corrector) — highest EV gap target
2. **F-3** (MV2H-driven ensemble selection) — uses item 6's sweep output
3. **F-2** (formant-band onset detector) — Vocadito gap
4. **F-5** (tempo-curve preservation) — structural Liszt
5. F-4, F-6, F-7, F-8, F-9, F-10 — opportunistic
