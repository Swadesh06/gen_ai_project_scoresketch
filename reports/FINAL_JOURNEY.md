# FINAL_JOURNEY.md — HumScribe project journey

## 1. Summary

HumScribe is an audio-to-score transcription system that turns a wave file
into a printable musical score (MusicXML + SVG). It handles two input
modes — *instrument* (piano or general) and *humming* (vocal) — through a
single six-stage pipeline, with an optional seventh arrangement stage.
After four engineering sessions (Phases A → G, 17/17 v4 work items
closed) the production system reaches **ASAP 9-piece MV2H = 0.6151**
(score beats) and **Vocadito A1 noff F1 = 0.6776** (mir_eval canonical),
the latter 6.4 pp shy of the human inter-annotator-agreement ceiling on
that corpus. Five generative-AI components are integrated alongside the
deterministic DP rhythm quantiser.

## 2. Goal and motivation

The course brief asked for a *generative-AI audio-to-score* system that
works for both instrumental and humming input, runs cross-platform, and
produces a real score (not a piano roll). The motivating use cases were
(a) sketching a melody by humming into a phone and (b) auto-transcribing
a piano recording for arrangement. Neither use case is solved end-to-end
in the literature on the corpora that ship public audio + score pairs
(ASAP for piano, Vocadito / MTG-QBH for humming, MAESTRO for sanity);
the project committed to honest evaluation on those corpora rather than
demo cherry-picking.

## 3. System architecture

```
                                       (Stage 2-A: instrument branch)
                                  ┌─────────────────────────────┐
                                  │  YourMT3+  /  ByteDance     │
                                  │  /  Basic Pitch             │
                                  │  (T5 seq2seq, MIDI tokens)  │
                                  └──────────────┬──────────────┘
                                                 │
 ┌──────────┐    ┌──────────┐    kind == piano   │     ┌──────────────┐
 │ Stage 0  │ ─→ │ Stage 1  │ ─→ instrument? ────┘ ─→  │  Stage 3     │
 │ audio I/O│    │ mode gate│                          │  normalise   │
 │ ffmpeg + │    │  humming │                          │ post_process │
 │ librosa  │    │/instr/   │ ─→ humming?  ─────────┐  │  G-4 merge,  │
 └──────────┘    │piano/guit│                       │  │  key + spell │
                 └──────────┘                       │  └──────┬───────┘
                                  ┌─────────────────┴──────┐  │
                                  │   PESTO pitch +        │  │
                                  │   CREPE periodicity    │  │
                                  │   (voicing gate)       │  │
                                  └────────────────────────┘  │
                                       (Stage 2-B: humming)   │
                                                              ▼
                          ┌─────────────┐   ┌──────────────┐   ┌──────────┐
                          │  Stage 4    │   │  Stage 5     │   │ Stage 6  │
                          │  beat       │ ─→│  rhythm DP   │ ─→│  render  │
                          │  beat_this  │   │  Cemgil-     │   │ music21  │
                          │  + F-1      │   │  Kappen      │   │ + Verovio│
                          │  octave     │   │  (tpb=12,    │   │  → SVG + │
                          │  sanity     │   │  per-voice)  │   │   MXL    │
                          └─────────────┘   └──────┬───────┘   └──────────┘
                                                   │
                                                   │  (Stage 7 optional)
                                                   ▼
                                         ┌────────────────────────┐
                                         │ arrange                │
                                         │ MusicGen-Melody-Large  │
                                         │ + C5b LoRA (JSB)       │
                                         └────────────────────────┘
```

Auto-routing: for Chopin-style piano (notes/sec < 10 AND pitch IQR < 24)
the pipeline activates the **B76 voice transformer** for voice IDs and a
per-voice DP path. See `humscribe/pipeline.py:_should_use_per_voice_dp`.

## 4. The five generative-AI components

1. **PESTO pitch + CREPE periodicity-as-voicing** (pretrained
   discriminative). PESTO 23 M parameters, CREPE 22 M. The Phase B+1
   insight was to use CREPE's *periodicity* channel (separate from its
   pitch head) as the voicing signal while keeping PESTO's pitch — +5.3
   pp on Vocadito A1 over PESTO-alone.
2. **YourMT3+** (~150 M-param T5 seq2seq autoregressive over MIDI
   tokens). Default piano transcriber for Romantic music. +6.1 pp ASAP
   9-piece snap over ByteDance, +12.6 pp on the 3-Romantic subset.
3. **MusicGen-Melody-Large** (3.3 B autoregressive over EnCodec audio
   tokens, melody-conditioned). The arrangement backend. Peak 6.25 GB
   VRAM on a 16 GB card thanks to fp16 + EnCodec's compact token grid.
4. **B76 voice transformer** (1.8 M-param 6-layer Transformer trained
   from scratch on 237 ASAP pieces). 94.5 % mean held-out voice-ID
   accuracy on the four Romantic test pieces; the per-voice DP that
   uses its output adds +1.66 pp on Chopin Berceuse.
5. **C5b r=64 LoRA adapter on JSB Chorales** (trained from scratch on
   315 four-part pairs, test loss 0.983 vs B77 distill baseline). The
   default arranger; F-7 chroma-similarity 0.689 over a raw-base 0.548.

These coexist with a *deterministic* Cemgil-Kappen rhythm DP at Stage 5
— the project's design stance is "generative where it pays, classical
where it doesn't."

## 5. Phase-by-phase journey

| phase | dataset(s) | headline | key wins | key honest discards |
|---|---|---|---|---|
| **A** (bootstrap) | Vocadito, MIR-1K, ASAP BWV 846 | Voc A1 0.538, BWV 846 snap 0.724 | initial pipeline, PESTO baseline | none yet — exploration |
| **B+1** (30+ exps) | + ASAP 5-Bach, MAESTRO sanity, MTG-QBH | Voc 0.665 (+12.7 pp), BWV 846 snap 0.847 (+12.3 pp) | B36 hybrid voicing, B15 voice tracking + per-voice DP, B16 VT hyperparams | B4/B6 HMM segmenter, B10/B19 BiLSTM onset (data-bound), B20/B25 HMM voice tracker |
| **B+2 / v3.4** | + ASAP Romantic, IAA study | Voc IAA ceiling 0.740 measured; Liszt structurally 0.078 | YourMT3+ swap-in (+12.6 pp Romantic), adaptive_pj, B58 confirms beat_this fine — ByteDance is the loss | B52 HuBERT BiLSTM (worse than B50), B54 tpb=48 Liszt rescue (oracle 0.155 — still bad), B59 basic_pitch drop-in (-25 pp) |
| **D** | + B76 train, B77 LoRA infra | B76 94.5 % mean voice acc, B77 LoRA 69 % loss decay, MusicGen-Large 6.25 GB peak | B76 voice transformer at scale, per-voice DP integration | B66 YMT3 on humming (-17 pp wrong domain), B69 MERT BiLSTM data-bound, B73 Transformer voicing |
| **E** (v3) | + MV2H wrapper | ASAP 9-piece MV2H **0.5492** (+0.022 from tpb=12 + F-1) | MV2H shipped, F-1 octave sanity (+0.088 BWV 856), tpb=24→12 default, C5b r=64 LoRA, Docker harness | item 2 MIR-ST500 stack (Voc 0.666 vs 0.69 target), item 6 sweep (+0.022 vs +0.03), item 7 ensemble (all members ≤ +0.01) |
| **G** (v4) | strict re-measurement | ASAP MV2H **0.6151** (+0.0636), Voc noff F1 **0.6776** (+0.0124) | G-1 voice emission (ASAP voice 0.825), G-2 meter emission (0.303), G-4 same-pitch merge | G-5 median smoothing (regresses 0.013), G-8 round-trip sign-inverted, G-11 render_tpb auto-detect (33 unreadable vs ≤5), 8 other discards |

## 6. Current headline numbers

| metric | value | source | reference |
|---|---|---|---|
| **ASAP 9-piece MV2H** (score beats, ymt3_cache, G-1+G-2 on) | **0.6151** | `reports/_item-g2.json` | +0.0636 from pre-Phase G 0.5515 |
| **Vocadito A1 noff F1** (40 clips, mir_eval canonical, G-4 on) | **0.6776** | `reports/_item-g4.json` | +0.0124 from pre-Phase G 0.6652 |
| Vocadito IAA ceiling | 0.740 | B51 study | human-to-human agreement on the same corpus |
| MAESTRO multi-pitch F1 (5-clip chamber, sanity) | 0.984 | B14 / PHASE_G | saturated |
| ASAP rhythm gate (Bach 846 stage-5 snap) | 0.847 | B12 / B16 | unchanged through G |

See `outputs/figures/F1_metric_trajectory.png` for the phase-by-phase
chart.

## 7. MV2H sub-axis breakdown (ASAP 9-piece, post-Phase G)

| sub-axis | value | comment |
|---|---|---|
| multi-pitch | **0.962** | saturated (YourMT3+ does this near-perfectly on ASAP cache) |
| voice | **0.825** | G-1 contribution — was 0.704 with `voices=[0]*n` |
| meter | **0.303** | G-2 contribution — was 0.103 with uniform tatum grid |
| value | **0.985** | saturated (post-DP duration quantisation is tight) |
| harmony | **0.000** | no chord-recognition module ships; explicit Phase H scope |

See `outputs/figures/F2_mv2h_subaxes.png`.

## 8. What worked and why

- **F-1 octave sanity** (Phase E): a notes-per-beat density + fast-tempo /
  slow-note rule halves or doubles beats when the inferred tempo is an
  octave wrong. Detector 9/9 on ASAP. Lift +0.088 MV2H on Bach BWV 856
  alone — the single biggest piece-level win the project has shipped.
- **tpb=12 default** (Phase E ME-14): 7/7 ASAP pieces prefer tpb=12 over
  tpb=24 with octave sanity on. +0.011 mean MV2H from the switch alone,
  and keeps an integer ratio to render_tpb=12.
- **G-1 + G-2 emission plumbing** (Phase G): writing real voice IDs (B76
  or single-voice fallback) into MV2H's input format, and writing the
  beat-interpolated tatum grid in place of a uniform one, accounts for
  the +0.0636 ASAP MV2H. Both are *emission-side* changes — they do not
  alter pitches or durations, just how the metric sees them.
- **G-4 same-pitch gap merging** (Phase G, CREPE Notes 2023 mechanism):
  on 40-clip Vocadito A1, precision moves +0.0325 while recall moves
  only −0.0134; net F1 +0.0124. The strict-pass winner.
- **MusicGen-Melody-Large at 6.25 GB**: arguably the most surprising
  finding — the 3.3 B model runs at the same wall as the 1.5 B because
  EnCodec autoregressive token count, not parameter count, is the
  bottleneck.

See `outputs/figures/F4_asap_per_piece.png` for the per-piece breakdown
and `outputs/figures/F5_g4_ablation.png` for the G-4 ablation.

## 9. What didn't, and why

Eight named discards (a sample of 30+ across the project):

1. **B47 voicing hysteresis**: +0.5 pp vs the +5 pp target. The voicing
   signal already binarises cleanly; hysteresis added latency without
   accuracy.
2. **B66 / DDSP humming→violin ensemble**: 0.484 vs 0.71 target. DDSP
   timbre transfer loses too much pitch information for downstream
   PESTO/CREPE.
3. **Item 2 MIR-ST500 pretrain** (Phase E): wrong domain — pop + backing
   data gave 0.30 test F1 on the transferred Vocadito target.
4. **B52 / B69 MERT and HuBERT learned voicing**: 40-clip Vocadito is
   too small for either feature stack to beat the 0.665 heuristic; the
   data ceiling, not the architecture.
5. **G-3 IOI octave detector**: Chopin Berceuse is at 3× tempo, halve
   /double cannot reach a 1/3 correction. 6/9 detector accuracy and
   zero lift on the target piece. See F-4 — Chopin Berceuse is the flat
   bar.
6. **G-5 median pitch smoothing** (Mauch 2014 pYIN 250 ms window): the
   strict gate measured −0.0132 noff F1 vs baseline on the canonical
   eval. Published window over-smooths against the segmenter family
   this pipeline uses.
7. **G-8 round-trip self-consistency**: distance correlates with MV2H
   in the *wrong direction* — Liszt has the lowest distance but is
   among the worst pieces. Note-count dominated the signal.
8. **G-10 bar-level diagnostic**: Liszt 0.49 vs the <0.4 selectivity
   threshold and BWV 846 0.50 vs the >0.8 — the diagnostic fails both
   tails simultaneously.
9. **G-11 render_tpb auto-detect**: 33 unreadable tuplets across four
   demos vs the ≤5 strict criterion. This is the regression W-1 of the
   present session reverts manually.
10. **Liszt structural ceiling**: the DP oracle is 0.132 — even with
    perfect voice assignment Liszt scores poorly because the rubato +
    dense chordal texture violates the DP's piecewise-constant tempo
    assumption.

Twenty more discards are recorded across `reports/exp_B*.md` and the
Phase G item files. The "no goalpost moving" rule means every discard
ships with both the original threshold and the observed value.

## 10. Honest limitations

- **Vocadito 6.4 pp below human IAA**: the heuristic + G-4 stack saturates
  at 0.6776 vs human agreement at 0.740. Closing the gap needs a larger
  labelled humming corpus, not a better algorithm on this one.
- **Offset detection ~22 pp below pitch detection**: Vocadito offset20
  F1 is 0.439 vs onset 0.6776. Humming has no analogue of a percussive
  attack; offset is the structural bottleneck of the whole humming side.
- **27 pp pre-F-1 gap between score-beat and real-beat ASAP MV2H**:
  before F-1 octave sanity, real-beat MV2H trailed score-beat by ~0.27.
  F-1 closes most of it for Bach; Romantic music still loses to beat-
  tracking error.
- **Harmony sub-axis = 0.000**: no chord-recognition module ships; the
  MV2H harmony component is left at floor.
- **Chamber voice tracking unfixed**: G-1's MAESTRO arm tops out at
  voice 0.488 across all three voice strategies (off / B76 / greedy).
  B76 was trained on piano left/right-hand, not multi-instrument; a
  chamber-trained tracker is Phase H.
- **Liszt out of scope**: the DP oracle (0.132) is the ceiling for the
  current rhythm formulation.

## 11. The disciplined process bits

- **Strict scorecard** (`reports/PHASE_G_STRICT_SCORECARD.md`): every
  Phase G item carries threshold + observed + pass/fail. 17/17 closed.
- **/goal commands and end-of-turn reporting** kept session focus on
  the spec items — no organic scope expansion.
- **"No goalpost moving" rule**: discards retain both the original
  threshold and the observed value. The 11 honest discards in Phase G
  are recorded with mechanism evidence (e.g. "Chopin needs 3× not 2×").
- **Parallelisation rules**: at most two GPU jobs concurrently, named
  tmux sessions, OOM protocol in `reports/_OOM_INCIDENTS.md`.
- **Regression checks**: every phase ends with the prior phase's gates
  re-run to verify non-targeted metrics stay within ±0.005.

## 12. Hardware and environment

- **GPU**: NVIDIA RTX 2000 Ada, 16 GB VRAM. All trained components fit
  with margin (largest peak was MusicGen-Melody-Large at 6.25 GB; B76
  voice transformer trains at < 1.5 GB).
- **conda env**: `humscribe` (Python 3.11). The session's pack/unpack
  workflow keeps the env reproducible across the four sessions.
- **Cross-platform**: a `Dockerfile` ships at the repo root; the
  audiocraft → HuggingFace `transformers.models.musicgen` swap means
  the MusicGen backend works on Windows where audiocraft fails.
- **Java**: MV2H is a Java jar called via `subprocess.run(timeout=...)`;
  the timeout is a deliberate guard against the documented hang on
  certain Vocadito clip pairs.

## 13. Key visual artifacts

- `outputs/figures/F1_metric_trajectory.png` — phase-by-phase ASAP MV2H
  + Vocadito noff F1 (this section's headline chart).
- `outputs/figures/F2_mv2h_subaxes.png` — five sub-axes pre/post G,
  showing the harmony floor explicitly.
- `outputs/figures/F3_strict_pass_distribution.png` — 17-item donut.
- `outputs/figures/F4_asap_per_piece.png` — 9 pieces pre/post with
  BWV 856 and Chopin Berceuse called out.
- `outputs/figures/F5_g4_ablation.png` — Vocadito G-4 vs G-5 vs combined,
  with the strict 0.67 line.
- `outputs/figures/F6_pipeline_architecture.png` — six stages + Stage 7,
  the visual cousin of the ASCII diagram in section 3.
- `outputs/figures/F7_demo_before_after.png` — 2×2 SVG comparison
  showing the MAESTRO chamber regression / W-1 revert and the Vocadito
  G-4 before/after.
- `outputs/demos/maestro_chamber3_30s.{svg,musicxml}` — clean chamber
  demo at render_tpb=8.
- `outputs/demos/maestro_chamber3_30s_phase_g_regression.{svg,musicxml}`
  — preserved regression evidence (tempo 154, 4× 24-lets + 1× 48-let).
- `outputs/demos/vocadito_1_humming_{before,after}.{svg,musicxml}` —
  G-4 ablation visual.

## 14. Phase H — future work (top 3)

1. **ME-6 chord recognition for the harmony axis**: lift the floor-0
   harmony sub-axis. Off-the-shelf chord-CNNs on Madmom-quality features
   could plausibly add +0.05–0.10 to ASAP MV2H without touching the
   pipeline core.
2. **Chamber voice tracker on MusicNet** (or MAESTRO-chamber MIDI):
   train a B76-class Transformer on 4-instrument supervision rather
   than 2-hand piano supervision. Closes the G-1 MAESTRO arm
   (target: voice 0.488 → 0.65+).
3. **Lakh MIDI LoRA training run** (G-13's incomplete arm): a real
   1–2 hour Lakh corpus prep + 3-soundfont render + LoRA fine-tune,
   then chroma-sim eval. The harness, dry-run logs, and OOM protocol
   are already in place.

## 15. Closing

HumScribe is finished — *honestly* finished. The 17-item Phase G scorecard
closes with 2 full strict passes, 2 partial, 11 documented discards, and
2 per-spec artifact closes; the headline numbers (ASAP MV2H 0.6151,
Vocadito noff F1 0.6776) are independently reproducible from the cached
features and the canonical eval scripts. The system shows what a small
generative-AI integration looks like when held to a non-negotiable
strict-pass discipline: a *deterministic* DP at the centre, five
*generative* components doing what they uniquely do, and a long list of
ideas that didn't survive the gate, each with a recorded reason. The
Phase H list above is what the project would do next, not a wish list of
ambitions; everything on it is bounded by the same kind of strict-pass
criteria as the work that shipped.
