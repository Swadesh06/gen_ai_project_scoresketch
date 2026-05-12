# task_description_v3.md — HumScribe Phase E

This is the spec for the next phase of HumScribe work. It builds on top of everything completed in Phase A, Phase B, Phase B+1, the v3.4 spec (`task_description_v2.md`), and the autonomous Phase D work (B65–B87b). Read those first if you haven't already.

## Important correction carried over from v2's evaluation

`results_v2_evaluation.md` flagged that my previous claim — "fine-tune `beat_this` on classical piano with score-aligned beats" as the highest-EV idea for the 27pp ASAP end-to-end gap — was wrong. `beat_this` is already trained on ASAP plus 14 other datasets including RWC Classical, JAAH, Filosax, and Hainsworth. The 27pp gap is not a data-poverty problem. The real fix is either (a) algorithmic post-processing of `beat_this`'s output (the agent's B87b `target_bpm=110` is a primitive version), or (b) accepting the gap and reporting both score-beat and real-beat numbers honestly.

**Do not attempt to fine-tune `beat_this` on ASAP.** It will not help.

## Phase E priority list

The eight work items below are ordered by impact-per-effort. Most can run in parallel — dependencies are called out explicitly.

| # | Work item | Resource class | Depends on |
|---|---|---|---|
| 1 | MV2H end-to-end score-similarity metric | **CPU only** | nothing |
| 2 | MIR-ST500 pretraining stack for the learned onset model | GPU (small) + CPU eval | nothing |
| 3 | DDSP humming→instrument-audio→transcription experiment | GPU (small) + CPU eval | nothing |
| 4 | Cross-platform Docker image | **CPU only** | items 5+6 ideally landed first |
| 5 | JSB Chorales real-pair training for B77 MusicGen LoRA | GPU (10 GB during train) | nothing |
| 6 | MV2H-driven hyperparameter sweep | **CPU only**, many parallel agents | item 1 must be done |
| 7 | Music-theory-guided ensemble members (14 candidates) | mostly **CPU only** | item 1 strongly recommended |
| 8 | Regenerate the MAESTRO chamber demo file | **CPU only** | nothing (one-liner) |

Plus a category of "future-ideation" items beyond the immediate Phase E — text-prompt style hints, tempo-curve preservation, score-conditioned LoRA, notation editor, demo-mode pre-baked hums, video-diff evaluation outputs. Documented at the end.

---

## Work Item 1 — MV2H end-to-end score-similarity metric (HIGHEST PRIORITY)

### Why

Every gate in the project measures one stage in isolation: PESTO RPA tests pitch tracking alone, `beat_this` F-measure tests beat tracking alone, snap-F1 tests rhythm quantization alone, COnP F1 tests note-level matching alone. **None of them captures "is the final score close to the score we wanted?"** The user's idea 1 from the conversation is to close this gap with a metric that compares the produced MusicXML to a ground-truth MusicXML.

This becomes:
- The headline metric for the course paper
- A hyperparameter-tuning reward signal (enables item 6)
- A structural debugging tool (the diff between two scores tells you exactly which notes diverged)

### What

Integrate **MV2H (Multi-pitch, Voice, Meter, and Harmony)** — Andrew McLeod's published metric for evaluating transcribed scores against ground-truth scores. Five sub-scores: multi-pitch detection accuracy, voice separation, meter detection, value (rhythmic duration) detection, harmony detection. Each in [0, 1], averaged for a single headline number.

Official Java implementation: `https://github.com/apmcleod/MV2H`. There are existing Python wrappers; if a clean one isn't available, write a thin Python wrapper that shells out to the Java jar and parses the output.

### How

1. Convert the produced MusicXML and ground-truth MusicXML to MV2H's text input format (a simple line-based representation of notes-per-beat). The conversion is mechanical — write `humscribe/eval/mv2h_io.py` with `score_to_mv2h_format(score: music21.stream.Score) -> str`.
2. Wrap the MV2H jar invocation: `humscribe/eval/mv2h.py` with `compute_mv2h(predicted_xml: str, reference_xml: str) -> dict` returning the five sub-scores plus the average.
3. Add `scripts/eval_mv2h_asap.py` that runs the metric over the 9-piece ASAP test set and reports per-piece + mean.
4. Add `scripts/eval_mv2h_vocadito.py` — but caveat: Vocadito doesn't ship reference MusicXML, only note-level annotations. Convert Vocadito annotations to a minimal MusicXML via music21 first.
5. Re-run on all existing baselines (B14 MAESTRO, B63 ASAP YourMT3+, B87b end-to-end ASAP, Vocadito A1/A2 hybrid). Log to WandB under tag `metric-mv2h`.

### Pass criteria

- MV2H metric runs without exception on 9 ASAP pieces + 40 Vocadito clips + 5 MAESTRO test tracks
- Per-piece MV2H scores logged to WandB and committed to `reports/_metric_mv2h_baselines.json`
- The structural diff is interpretable: when MV2H drops on a piece, the per-axis sub-scores reveal whether the failure was pitch, voice, meter, or rhythm
- Report `reports/item-1_mv2h_metric.md` includes:
  - MV2H baseline numbers across the entire eval set
  - Correlation analysis: how does MV2H correlate with note-F1 / snap-F1 / COnP-F1? (use Pearson + Spearman)
  - At least one example of a "MV2H said this is worse" piece where note-F1 looked equivalent, demonstrating the metric catches something other metrics miss

### Compute footprint

CPU-only. MV2H jar runs in ~50 ms per piece. The whole eval-set re-run takes minutes. Java required on the server (`apt install default-jre`).

### Co-scheduling

Trivial — runs alongside anything. Spin up the eval as a tmux session while any GPU experiment is in progress.

### Why this is first

Every subsequent Phase E experiment will produce outputs that need to be evaluated. Without MV2H, you'd be measuring snap-F1 or note-F1 — which the agent has already pushed close to their stage-wise ceilings. MV2H is the new objective. Building it first means items 2, 3, 5, 6, 7 all get evaluated against it from day 1.

---

## Work Item 2 — MIR-ST500 pretraining stack for the learned onset model

### Why

The agent's seven attempts at a learned voicing/onset detector for humming (B10, B19, B42, B50, B52, B69, B73) all plateaued below the heuristic voicing-threshold baseline. The structural reason was data quantity: 40 Vocadito clips is insufficient for any learned model. MIR-ST500 has **500 pop songs (~30 hours)** with hand-annotated onset+offset+pitch. That's a 10× data increase — typically the threshold at which BiLSTMs start beating heuristics.

### What

A three-stage training pipeline for an onset/voicing model:

1. **Pretrain** on DALI v2 (7,756 songs, semi-automatic annotations) — representation learning
2. **Fine-tune** on MIR-ST500 (400 train songs, hand-labeled) — accuracy calibration
3. **Final fine-tune** on Vocadito (40 clips, 5-fold CV) — casual-humming distribution match

### How

- **Datasets**: MIR-ST500 (from `github.com/york135/singing_transcription_ICASSP2021`, the `get_youtube.py` script fetches audio); DALI v2 (`github.com/gabolsgabs/DALI`); reuse existing Vocadito loader. Download budget: ~50 GB total disk after audio fetch.
- **Architecture**: 4-layer BiLSTM on mel-spectrogram features at 10 ms hop, hidden size 128, dropout 0.3. Inputs: 80-bin mel-spectrogram, 16 kHz audio. Outputs: per-frame voicing logit + per-frame onset logit. Match the agent's B19 architecture so any improvement is attributable to data not model.
- **Loss**: weighted BCE on both heads; weight onset class higher (class imbalance ~50:1).
- **Schedule**:
  - DALI pretrain: 5 epochs, LR 3e-4, AdamW, batch 16, sequence length 10 s.
  - MIR-ST500 fine-tune: 20 epochs, same hyperparameters, sequence length 30 s.
  - Vocadito fine-tune: 5-fold CV, 10 epochs per fold, LR 5e-5.
- **Mandatory unit tests before launching long runs** (carry forward ScoreSketch tombstones):
  - Single-batch overfit: loss < 0.05 on 1 batch within 100 steps. Catches double-BOS-style bugs.
  - Warmup-vs-total-steps assertion: `warmup_steps < 0.1 * total_steps`. Catches the v3.4 warmup-overrun tombstone.
  - Inference smoke test: run the model on a 5-s sine and assert non-trivial output before training starts.

### Pass criteria

- Final Vocadito A1 no-offset F1 ≥ 0.69 (clearly above the heuristic 0.665 — 2σ above the agent's cross-validation noise)
- MV2H on Vocadito clips improves by ≥ 0.02 over the heuristic baseline (item 1 must be built first to evaluate this)
- Inference latency per 30-s clip < 200 ms on GPU

### Decision rule

If Vocadito A1 ≥ 0.69 AND MV2H delta ≥ +0.02: promote as the default voicing/onset backend for the humming branch, ablating the heuristic to a `--voicing-backend heuristic` flag for legacy. If either misses: leave behind a `--voicing-backend bilstm_mirst500` flag and write the decision rationale in the report.

### Compute footprint

- DALI pretrain: ~15 GB peak audio I/O cache; ~3 GB VRAM during training; many hours wall on a single GPU
- MIR-ST500 fine-tune: ~3 GB VRAM
- Vocadito fine-tune: <1 GB VRAM

Dominated by I/O and audio decoding. **The CPU side runs the audio decoder in parallel with GPU training** — the dataloader's `num_workers` should be set so that GPU never waits for data (use `nvidia-smi dmon` to verify).

### Co-scheduling

GPU is busy ~6 GB of 32 GB during all three stages. Co-schedule:
- The MV2H eval (item 1) on CPU while training runs
- Dataset preprocessing for next items (DDSP audio prep for item 3) on CPU in another tmux
- The Docker build (item 4) on a CPU container while training runs

---

## Work Item 3 — DDSP humming→instrument-audio→transcription experiment

### Why

The user's idea 2 from the conversation: if humming-to-score is hard, route the hum through a humming-to-instrument-timbre-transfer model first, then transcribe the clean instrument audio through the much-more-accurate instrument pipeline. Worth a focused experiment that either rules out a popular hypothesis or opens an architectural change.

### What

A new branch `humscribe/pitch/timbre_transfer/` that takes a hummed audio clip → DDSP (Magenta) timbre-transferred audio in a target instrument (violin) → that audio runs through the existing **instrument** transcription path (PESTO + standard segmenter + DP), bypassing the humming path entirely.

### How

- **Model**: Magenta's DDSP (Differentiable DSP) with a pretrained violin checkpoint. Magenta ships pretrained models for violin, flute, trumpet, saxophone. **Start with violin** — its harmonic content most closely matches a sung melody. https://github.com/magenta/ddsp
- **Pipeline integration** in `humscribe/pipeline.py`:
  ```python
  if cfg.humming_via_timbre_transfer:
      audio = ddsp_humming_to_violin(audio)
      cfg.input_kind = "instrument"
      cfg.instrument_hint = "violin_synthetic"
  # continue with instrument-path Stage 2-A
  ```
- **Variants to test**:
  - Pure DDSP→instrument path
  - **Ensemble version**: run BOTH the existing humming path AND the DDSP→instrument path, then per-note majority vote (weight by each path's confidence — PESTO confidence on direct, instrument-pipeline confidence on transfer path). This is the highest-EV variant: it tests whether the two paths have uncorrelated failure modes, which is the only condition under which ensembling works (the agent's prior B3/B11/B17/B27 failed precisely because they ensembled correlated models).

### Pass criteria

- Direct DDSP path Vocadito A1 ≥ 0.55 (acceptable but probably worse than 0.665 baseline)
- **Ensemble path Vocadito A1 ≥ 0.71** (this is the win condition — beats both individual paths because the failure modes are uncorrelated)
- MV2H delta of ensemble vs baseline ≥ +0.03 (item 1 must be built)
- Latency overhead ≤ 5 s per 30-s clip (DDSP runs at ~10× realtime on CPU)

### Decision rule

If ensemble Vocadito A1 ≥ 0.71 AND MV2H ≥ +0.03: ship as `--humming-backend hybrid_ddsp` flag, default-off. Promote to default only after a multi-clip user-recording test confirms the win generalizes.

If only direct path improves: ship as flag, document.

If neither improves: write a clean "negative result" report. This is informative — it's the kind of approach often suggested in the literature without rigorous evaluation.

### Compute footprint

- DDSP inference: ~1 GB VRAM, ~5 s per 30-s clip on GPU (or ~30 s on CPU)
- Existing instrument pipeline: ~5 GB VRAM
- Total: ~6 GB VRAM peak

### Co-scheduling

CPU work alongside: MV2H eval, Docker build, dataset preprocessing.

GPU work alongside: small — DDSP doesn't compete with most GPU work. Could co-locate with item 5 (LoRA training) if VRAM allows after checking peaks.

---

## Work Item 4 — Cross-platform Docker image

### Why

The user wants the system installable on any OS including Windows. FluidSynth, `audiocraft`, and `pyfluidsynth` all have Windows pain. Docker sidesteps the OS-specific dependency problem.

### What

A Dockerfile that produces a single image users can pull and run. Includes the full audio toolchain, all pretrained model weights pre-cached, and an entrypoint that launches the Streamlit app on port 8501.

### How

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

# system audio deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential fluidsynth fluid-soundfont-gm sox ffmpeg \
    libsndfile1 libsox-fmt-all default-jre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pre-install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-cache pretrained weights so first run doesn't block
RUN python -c "import torchcrepe, torch; \
    torchcrepe.predict(torch.zeros(1, 16000), 16000, 160, model='full', device='cpu')" \
    && python -c "from pesto import load_model; load_model('mir-1k_g7', step_size=10.0)" \
    && python -c "from piano_transcription_inference import PianoTranscription; PianoTranscription(device='cpu')" \
    && python -c "from beat_this.inference import File2Beats; File2Beats(checkpoint_path='final0', device='cpu')"

COPY . .
RUN pip install -e .

EXPOSE 8501
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

Additional moves:

- **Replace `audiocraft` with HuggingFace `transformers.models.musicgen`** before the Docker build. `audiocraft` is brittle on Windows; the HF implementation uses the same weights and has clean cross-platform support. Update `humscribe/arrange/musicgen.py` accordingly. This change is also valuable independent of Docker.
- **Verify `pyfluidsynth` (not `fluidsynth`)** in `requirements.txt` — these are different packages, the latter is unrelated.
- **Pre-cache YourMT3+ and MusicGen-Melody-Large weights** in the Docker build for instant first-call startup. Adds ~15 GB to the image but eliminates a long first-run download. Build with `--squash` to keep the layer size manageable.

### Pass criteria

- `docker build -t humscribe .` succeeds end-to-end without manual intervention
- `docker run -p 8501:8501 humscribe` launches the Streamlit app reachable on `localhost:8501`
- All six demo workflows work inside the container: piano transcribe, humming transcribe, mode soft/medium/hard, arrangement generation, MIDI download, SVG render
- Image size ≤ 8 GB without pre-cached weights, ≤ 25 GB with all weights pre-cached

### Compute footprint

CPU-only during build, but the build downloads ~15 GB of pretrained weights, so network bandwidth is the practical bottleneck.

### Co-scheduling

Runs in background as a long-running build job. Use a dedicated tmux session `docker-build`. Doesn't interfere with anything else.

---

## Work Item 5 — JSB Chorales real-pair training for B77 MusicGen LoRA

### Why

B77 demonstrated the MusicGen LoRA fine-tuning pipeline works (69% loss decay, 4.72M trainable params at r=32, 8.57 GB VRAM peak), but the training data was 6 distill pairs synthesized by MusicGen itself — the adapter memorized 6 specific outputs and doesn't generalize. To turn B77's infrastructure into a useful artifact, train on real (melody, arrangement) pairs.

JSB Chorales is the cleanest source: 382 Bach four-voice chorales as MIDI, soprano line as the melody, full four-voice as the arrangement.

### What

- Render JSB Chorales MIDI files: soprano alone as melody audio (flute SoundFont via FluidSynth), full four-voice as arrangement audio (organ or string-quartet SoundFont). 382 pairs.
- Train MusicGen-Melody LoRA on these pairs with chromagram melody conditioning matching the audio synthesis chain.
- Compare against the B77 baseline (synthetic distill pairs).

### How

- **Data prep** (CPU-bound, parallelizable):
  - Download JSB Chorales MIDI: `github.com/cuthbertLab/music21` ships it; alternative: jsbach.net.
  - Render with `pretty_midi.fluidsynth()` using two SoundFonts: `flute.sf2` for melody, `organ.sf2` or `strings.sf2` for arrangement. Use FluidR3_GM as the default.
  - Trim each pair to a consistent length (15 s) and normalize loudness.
  - Save as `(melody.wav, arrangement.wav, melody_chroma.npy)` triples.
- **Training**:
  - Same LoRA architecture as B77: r=32, target attention modules of MusicGen-Melody-Large.
  - 1000 training steps with batch size 4 (gradient accumulation if needed for VRAM).
  - AdamW lr=1e-4, cosine schedule, **warmup over 100 steps not 1000**.
  - Same mandatory unit tests as in item 2 (single-batch overfit, warmup-vs-total assertion, inference smoke).
- **Evaluation**:
  - 50-pair held-out test set
  - Subjective: 5 human listeners rate "did the arrangement follow the melody" on a 1-5 scale
  - Objective: chroma-similarity between melody input and generated output

### Pass criteria

- Training completes without OOM
- Held-out test loss < B77's baseline final loss
- Subjective melody-following score ≥ 3.5/5 on average
- Generated arrangements do NOT just play the melody as flute (a failure mode where LoRA over-conditions on the input)

### Decision rule

If pass: replace B77 default adapter with the new one in `humscribe/arrange/musicgen.py`. The B77 adapter remains available as `--lora-adapter b77_distill` for ablation.

If fail: log negative result. The fallback is the user's existing chroma-only conditioning without LoRA.

### Compute footprint

- Data prep: pure CPU, parallelizable across cores
- LoRA training: ~10 GB VRAM peak (matches B77's measured 8.57 GB plus some headroom)
- Evaluation generation: ~13 GB VRAM (full MusicGen-Melody-Large)

### Co-scheduling

- Data prep (CPU) runs in parallel with item 2 GPU training
- LoRA training (10 GB VRAM) co-locates with item 1 MV2H eval (CPU) and item 4 Docker build (CPU)
- Cannot co-locate with another large GPU model — MusicGen takes the full GPU during training

---

## Work Item 6 — MV2H-driven hyperparameter sweep

### Why

The agent has been running WandB Sweeps optimizing per-stage F1 metrics. With MV2H built (item 1), you can optimize the actual end-to-end objective instead of stage-wise proxies. Many DP parameters, voicing thresholds, and tatum-resolution choices are currently set to values that maximize individual gate metrics but may not maximize end-to-end score quality.

### What

A WandB sweep over the pipeline's heuristic parameters using MV2H as the optimization target.

### How

- **Parameter space**:
  - `tatums_per_beat ∈ {6, 12, 24}` (currently 24)
  - `tatums_for_render ∈ {6, 12, 24}` (currently 12)
  - `complexity_alpha ∈ [0.5, 3.0]` (DP complexity penalty)
  - `sigma_quant ∈ [0.02, 0.06]` (DP quantization error scale)
  - `voicing_psw ∈ {13, 15, 17, 19, 21}` (humming voicing patience)
  - `voicing_vt ∈ [0.65, 0.85]` (humming voicing threshold)
  - `target_bpm_correction ∈ [80, 130]` (B87b tempo-octave fix center)
  - `tpb_allowed_denoms` — 3 settings: {1,2,3,4,6,8,12,16}, {1,2,3,4,6,8,12,16,24}, {1,2,3,4,6,8}
- **Eval set per sweep run**: 5 ASAP pieces (mixed difficulty) + 10 Vocadito clips. Small enough to fit each sweep run in <5 min wall time.
- **Sweep strategy**: Bayesian, 100 runs.
- **Many parallel agents**: launch ~6 sweep agents in parallel tmux sessions. Each agent is CPU-only (DP runs are CPU), so you can saturate the cores.

### Pass criteria

- Sweep converges within 100 runs (no improvement in the last 20 runs at the top of the leaderboard)
- Best configuration MV2H ≥ current-default MV2H + 0.03 on the held-out test set
- Best config no-regression on the worst-performing individual piece (no "win on average, regress on Liszt")

### Compute footprint

CPU-only. Each sweep agent uses ~1 core for DP plus PESTO/CREPE/ByteDance inference. Cache PESTO/CREPE/ByteDance outputs to disk once per piece, then each sweep agent reads cached features → pure CPU → fits as many agents as you have cores.

The cached-features path is the right one. Adds ~2 hours of one-time preprocessing, then 6 sweep agents can run on CPU with no GPU competition.

### Co-scheduling

Once features are cached, this is the ultimate parallelization-friendly workload: 6+ CPU agents, the GPU is free for whatever else. Use it as the "filler" workload whenever a GPU job is running solo.

---

## Work Item 7 — Music-theory-guided ensemble members

### Why

The agent's prior ensemble attempts (B3, B11, B17, B27) all failed because they averaged correlated estimators (PESTO + CREPE + ensembles thereof — same training data, same input features, same failure modes). **B36's hybrid voicing is the existing proof that uncorrelated ensembling works**: PESTO for pitch + CREPE periodicity for voicing → +5.3pp.

The 14 ensemble members below are designed for legitimate uncorrelated participation — each has either a different inductive bias, different input modality, theoretical correctness in a specific regime, or different training distribution.

**Most ensemble work targets bias correction, not variance reduction.** The pipeline's failures aren't noisy; they're systematically biased (the DP picks the wrong tuplet denominator on certain rhythm patterns, `beat_this` puts beat 1 at the wrong place when there's an anacrusis). Music theory tells you *which way* the bias goes, so a theory-derived corrector applies the right counter-bias. That's why these can outperform the agent's prior statistical ensembles even with fewer members.

A thoughtful ensemble of ~5 of these could plausibly move Vocadito from 0.665 to 0.70–0.72 and ASAP real-beat from 0.506 to 0.58–0.65. The agent should pick which to integrate based on the failure analysis in `results_v2_evaluation.md` and the gap targets in this spec.

### The 14 candidate members, organized by stage

**Pitch tracker / segmenter (gap 2):**
- **ME-1: pYIN as a non-neural diversifier** — pure DSP, different failure modes from neural trackers. Ships with librosa, CPU-only. Vote with PESTO+CREPE in regions of disagreement.
- **ME-2: Goto spectral-template pitch tracker** — cross-correlate audio spectrum with harmonic templates at p, 2p, 3p... weighted 1/n. Pure DSP, no training. Strong on clean tonal sounds where neural trackers can hallucinate.
- **ME-3: SwiftF0** — 96k-param CNN, 42× faster than CREPE, joint cls+reg head. Different output distribution than PESTO. Minimal compute, uncorrelated vote.

**Rhythm/quantization (gap 1, gap 3):**
- **ME-4: Tonal-meter prior on DP** — music theory says strong scale degrees (tonic, dominant) preferentially land on strong beats. Build a small prior over (beat-position × scale-degree) pairs from a MusicXML corpus (Bach Chorales, Lakh MIDI). Tie-breaker for the DP. **One of the highest-EV members on this list.**
- **ME-5: Phrase-boundary detection** — local maxima of inter-onset-interval signal a phrase boundary. Pure signal processing. Adds bar-line hint to the DP.
- **ME-6: Harmonic-rhythm prior** — chord changes typically land on strong beats. Cheap chord recognition (template match against 24 major/minor triads) → vote on beat positions. Polyphonic-instrument-only.
- **ME-7: Pickup-note / anacrusis detection** — first note in <300 ms before a "stronger" event is a pickup. Beat 1 is then offset by one note. Addresses a known `beat_this` failure mode. Cheap heuristic. **High legitimacy.**

**Score construction (visual quality):**
- **ME-8: Spiral-array key estimation** — Sapp's voice-leading geometry method. Ensemble with KrumhanslSchmuckler. When the two agree → confident, when they disagree → report uncertainty in the rendered score.
- **ME-9: Line-of-fifths enharmonic spelling** — Temperley 2001. Spell each note so the interval to the next note has minimum letter-name distance. Pure visual-quality win, doesn't move F1. **Cheap, polish-level fix.**
- **ME-10: Meter-template ensemble** — run 5 time-signature hypotheses (2/4, 3/4, 4/4, 6/8, 12/8) through the DP independently, pick whichever has lowest total DP cost. Embarrassingly parallel.

**Onset/offset (gap 2, gap 3 humming-side):**
- **ME-11: Formant-band onset detector** — onset detection on the 1.5-3.5 kHz formant band, where vocal-tract noise is weaker than in the fundamental band. Different uncorrelated signal from voicing-based onsets. **High legitimacy for humming.**
- **ME-12: Phase-deviation onset detector** — spectral phase derivatives (Bello et al. 2005). Different feature space than the agent's magnitude-based features. Available in librosa.

**Voice tracking:**
- **ME-13: Voice-leading legality score** — rule-based "a voice rarely jumps > major 6th; voices rarely cross" prior. Combine with B76's output. Low marginal value because B76 is already at 94% mean.

**End-to-end:**
- **ME-14: MV2H-driven system-level ensemble selection** — run N different pipeline variants (different pitch backend, different DP params, different time-signature hypothesis), pick the variant whose output minimizes MV2H. **Depends on item 1. The strongest theoretical move on this list — using the actual end-to-end objective to choose between candidates.**

### Recommended integration order

Build in this sequence (each member is an independent experiment with its own report):

1. **ME-9** (line-of-fifths spelling) — visual polish, no F1 risk
2. **ME-4** (tonal-meter prior on DP) — addresses ASAP real-beat gap directly
3. **ME-11** (formant-band onset detector) — addresses Vocadito offset gap
4. **ME-7** (anacrusis detection) — addresses a specific known failure mode in beat tracking
5. **ME-10** (meter-template ensemble) — addresses wrong-time-signature failures, embarrassingly parallel
6. **ME-1** (pYIN diversifier) — adds an uncorrelated pitch vote
7. **ME-14** (MV2H system-level ensemble) — requires item 1, but it's the strongest move

Skip: ME-3 (incremental), ME-6 (chord recognition is a sub-research-project), ME-13 (B76 is already near ceiling).

### Pass criteria per member

Each member must:
- Improve MV2H by ≥ +0.01 on the relevant eval slice (instrument or humming)
- Not regress any individual piece's MV2H by more than 0.02
- Not regress note-F1 or COnP-F1 by more than 1pp

If a member doesn't meet these, document as a negative result and discard.

### Pass criteria for the full ensemble

After integrating the kept members:
- Vocadito A1 noff F1 ≥ 0.70 (was 0.665, ceiling 0.740)
- ASAP real-beat MV2H mean ≥ baseline + 0.05
- All ensemble members run within the existing inference time budget (≤ 30 s per 30-s clip end-to-end)

### Compute footprint

12 of 14 members are CPU-only. ME-3 is tiny GPU (<0.5 GB). ME-6 is small GPU (~1 GB). The full ensemble adds < 1.5 GB GPU plus modest CPU work.

### Co-scheduling

This is ideal CPU work to run in parallel with any GPU experiment. The MV2H sweep (item 6) and the ensemble member integrations (item 7) are both CPU-bound and can co-run with items 2, 3, 5 (GPU work).

---

## Work Item 8 — Regenerate the MAESTRO chamber demo file

### Why

`results_v2_evaluation.md` flagged that the agent's rendering polish (item 1 of v3.4) updated three of the four demo SVGs but not `demos/maestro_chamber3_30s.svg`. The file is still showing `♩ = 73.17073170731705`, no key signature, 9× 24-lets, 3× 48-lets — the pre-polish output.

### What

One CLI call:

```bash
python -m humscribe.cli transcribe \
    outputs/maestro_clips/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1_30s.wav \
    --kind instrument --instrument piano --mode soft \
    --out outputs/demos/maestro_chamber3_30s
```

Then visual-diff the output against ground truth, commit, push.

### Pass criteria

The new `outputs/demos/maestro_chamber3_30s.svg` has integer tempo display, a key signature, and zero 24-lets or 48-lets.

### Compute footprint

A single inference pass. ~10 s. Trivial.

### Co-scheduling

Drop-in anywhere.

---

## Future-ideation items (Phase E+1 candidates)

These are documented for context but **not part of Phase E's commit-by deliverables**. Pursue if and only if items 1–8 land cleanly with time remaining.

- **Text-prompt style hints** — free-text "jazz waltz" / "jig" / "ballad" hints from the user, embedded via MiniLM, biasing the DP's complexity prior. Lightweight, interpretable.
- **Tempo-curve preservation** — stop averaging `beat_this`'s IBI to a single BPM; feed the local IBI sequence into the DP so it respects rubato. Cemgil-Kappen DP already supports this.
- **Score-conditioned LoRA for MusicGen** — pass the transcribed MIDI as a second conditioning input alongside chromagram. Strictly more information.
- **Web-based notation editor** — let users correct transcription mistakes in the rendered SVG. Each correction logs a (audio, correct-notation) pair to a flywheel database. Long-term play: 100 corrected hums beats the current Vocadito 40 by 2.5×.
- **Pre-baked example hums in the Streamlit demo** — 5–10 ready-to-load Twinkle/Mary/etc. recordings. Tiny work, big UX win.
- **Video diffs as evaluation outputs** — side-by-side scrolling rendered score vs ground-truth score synced to audio playback. Better course-paper figure than any F1 number.

---

## Cross-cutting requirements

These apply to every Phase E work item.

**Evaluation discipline:**
- Every kept change must report both MV2H and the corresponding stage-wise metric (note-F1, snap, COnP, etc.). MV2H alone could mask stage-level regressions; stage metrics alone miss end-to-end quality.
- Every change that affects rendered output must include a side-by-side SVG diff in its report. The agent's prior session missed this on item 1 (rendering polish) and the agent's report claimed completeness while the MAESTRO file went unchanged.
- ASAP numbers always cited with their beat source. "MV2H = 0.X (score beats)" or "MV2H = 0.X (real beats from beat_this)". The 27pp gap is real; the headline number deserves the footnote whenever it appears.

**Process discipline:**
- Single-batch overfit unit test before every training run. Catches double-BOS, warmup-overrun, dataloader bugs at low cost.
- Warmup-vs-total-steps assertion in every train script. `warmup_steps < 0.1 * total_steps`.
- Inference smoke test after every checkpoint save.
- WandB tag `phase-e` on every Phase E run for dashboard filtering.

**Decision rule defaults:**
- "Promote to default" requires beating the previous default on both MV2H AND at least one stage-wise gate metric, on the held-out eval set.
- "Keep behind flag" is the fallback when a method works on a subset but doesn't generalize.
- "Discard" is the fallback when nothing wins — write the negative-result report and move on.

**Honest reporting:**
- Half the agent's prior experiments produced negative results. Continue that discipline. The agent's PHASE_B and PHASE_D summaries are exemplary models for how to write honest cumulative-progression reports.
- If a Phase E work item misses its pass criteria, write the report saying so, do not move goalposts, do not avoid the result by re-defining the metric.

---

## End of task_description_v3.md

Apply on top of v2 and the agent's Phase D code. Architecture for the existing six stages plus optional Stage 7 is unchanged. Phase E adds:

- One new evaluation metric (MV2H)
- One new training pipeline (MIR-ST500 stack)
- One experimental architecture branch (DDSP timbre-transfer)
- One deployment artifact (Docker image)
- One LoRA retraining run (JSB Chorales)
- One end-to-end hyperparameter sweep
- A library of 14 ensemble members (5–7 to be integrated based on the failure analysis)
- One cleanup task (MAESTRO demo regen)

The architecture story expands to: *"a hybrid pretrained-first pipeline with discriminative stages for bounded subproblems, generative stages for open-ended generation, an end-to-end score-similarity objective for hyperparameter tuning, and a library of music-theory-guided ensemble correctors targeting specific structural failure modes identified in the failure analysis."*
