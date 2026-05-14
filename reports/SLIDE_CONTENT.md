# SLIDE_CONTENT.md — 4-minute presentation source

Four slides; speaker notes 40–60 words each; ≤ 800 total words.

---

## Slide 1 — Problem Statement

**Title**: Audio → Musical Score

**Bullets**

- Transcribe any audio (piano or humming) to standard music notation.
- Three modes: soft (audio only), medium (+ BPM), hard (+ key + meter).
- Output: MIDI / MusicXML / SVG / printable score.
- Cross-platform deployable: Linux, macOS, Windows via Docker.

**Figure**: `outputs/figures/S1_concept.png`

**Speaker note**
We frame this as a generative-AI problem with two distinct input modalities:
clean instrument and casual humming. Both are audio, but their failure modes
differ — instrument transcribers stumble on humming vibrato, humming
transcribers undershoot rhythmic precision. One pipeline serves both via a
mode gate, then re-converges before the score-rendering stage.

---

## Slide 2 — Existing Work Gaps

**Title**: Per-Axis Headroom Is the Real Story

**Bullets**

- AMT papers report single F1 scores that mask which axis is failing.
- MV2H (McLeod 2018) breaks evaluation into five sub-axes.
- Multi-pitch and value are saturated; voice, meter, harmony have headroom.
- Humming corpus IAA ceiling is 0.740 — humans disagree on note boundaries.
- Heuristic rhythm quantizers fail on rubato and slow input.

**Figure**: `outputs/figures/S2_subaxis_headroom.png`

**Speaker note**
The biggest wins came from realizing the pipeline already produced voice and
meter information (B76 voice tracker, DP tatum positions) but threw it away
at the MV2H emission boundary. Plumbing existing outputs into the metric text
format lifted ASAP MV2H by +0.0636 without changing transcription. Visibility
came before optimization.

---

## Slide 3 — Proposed Method

**Title**: Six-Stage Pipeline + Optional Generative Stage 7

**Bullets**

- Stages 0–6: deterministic backbone (I/O, mode, transcribe, normalize, beat, DP, render).
- Stage 2 splits by input kind: YourMT3+/ByteDance vs PESTO+CREPE+HMM.
- B76 voice transformer trained from scratch, 94% mean ASAP accuracy.
- Stage 7 optional: MusicGen-Melody-Large + C5b r=64 LoRA arrangement.
- Five generative-AI components: three pretrained, two trained from scratch.

**Figure**: `outputs/figures/S3_pipeline.png`

**Speaker note**
Hybrid architecture — discriminative pretrained components for bounded
subproblems, autoregressive generation for open-ended arrangement,
deterministic DP for rhythm quantization. The two trained-from-scratch
components (B76 voice transformer on 237 ASAP pieces, C5b LoRA adapter on
JSB Chorales) are the original contributions of this project.

---

## Slide 4 — Results

**Title**: Headline Numbers

**Bullets**

- ASAP 9-piece MV2H: **0.6151** (+0.0636 from Phase G emission work).
- Vocadito A1 noff F1: **0.6776** (+0.0124 from G-4 merging), vs IAA 0.740.
- MAESTRO multi-pitch F1: 0.984 (saturated).
- Phase G: 2 full strict passes + 2 partial + 11 honest discards + 2 artifacts.
- 30+ negative results documented honestly across four sessions.

**Figure**: `outputs/figures/S4_before_after.png`

**Speaker note**
G-4 same-pitch merging (CREPE Notes 2023 published practice) is the cleanest
single intervention on humming. Our strict-ablation discipline caught that it
must ship alone — the initial bundled deployment with G-5 and G-6 destroyed
its gains. The visible before/after shows 13 triplet brackets dropping to
zero on the same melody.

---

## Closing line (timer permitting)

The full journey including 30+ named negative results, the strict scorecard,
and architectural decisions is in `reports/FINAL_JOURNEY.md`. Visualizations
are in `outputs/figures/F1`–`F7` (long form) and `S1`–`S4` (this deck).
