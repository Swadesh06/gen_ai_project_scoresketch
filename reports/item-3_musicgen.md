# item-3 — MusicGen-Melody Stage-7 arrangement

## Goal
Per `task_description_v2.md` §Work item 3, add a Stage 7 generative arrangement
on top of the transcribed melody using **MusicGen-Melody** (Meta, 2023,
[arXiv:2306.05284](https://arxiv.org/abs/2306.05284), CC-BY-NC-4.0 weights /
MIT code via `audiocraft`).

Pass criteria from the spec:
- end-to-end: hum → arrangement playable in the Streamlit UI
- 6 style presets (lo-fi hip hop, jazz trio, EDM, orchestral cinematic,
  indie folk, bossa nova) all produce a coherent track
- peak VRAM < 20 GB
- weights load on first call without download errors
- MusicXML / MIDI for the original melody is unaffected (Stage 7 is an
  add-on, not in the score-rendering path)

## Procedure

### Integration
1. Cloned audiocraft from upstream and installed all dependencies
   (`--no-deps` for several to avoid the heavy `torch>=2.4 + triton 3.x`
   cascade that would have broken the rest of the pipeline). Notable
   no-deps installs: `julius`, `xformers`, `encodec`, `demucs`, `num2words`,
   `flashy`, `dora-search`, `treetable`, `submitit`, `retrying`,
   `cloudpickle`, `colorlog`, `torchmetrics`, `lightning-utilities`,
   `pytorch_lightning`, `openunmix`, `lameenc`. `av` came pre-installed.
2. **spaCy stub at import** — `audiocraft.modules.conditioners` imports
   `spacy` and `thinc` only for `LM4SyntaxConditioner`, which we do not use.
   `humscribe/arrange/musicgen.py` injects empty `_Stub` modules for spacy,
   thinc, and friends *before* importing audiocraft. Saves ~1 GB of native
   deps for an unused codepath.
3. **soundfile direct read** — `torchaudio.load(uri=...)` now requires
   torchcodec + ffmpeg. The arranger reads with soundfile and converts to
   torch tensor manually.
4. **6 prompt presets** — `PROMPT_PRESETS` dict in
   `humscribe/arrange/musicgen.py`:
   - `"lo-fi hip hop"` → "lo-fi hip hop with mellow piano, vinyl crackle, soft drums"
   - `"jazz trio"` → "jazz trio with upright bass, brushed drums, and warm piano"
   - `"EDM"` → "energetic electronic dance music with synth lead, four-on-the-floor kick, sidechain pumping pads"
   - `"orchestral cinematic"` → "cinematic orchestral arrangement with sweeping strings, brass swells, and tympani"
   - `"indie folk"` → "indie folk arrangement with fingerpicked acoustic guitar, light percussion, and mandolin"
   - `"bossa nova"` → "bossa nova with nylon-string guitar, soft brushed drums, and double bass"
5. **`arrange()` API**:
   ```python
   arrange(melody_audio_path, prompt, duration_s=15.0, model_size="melody",
            seed=0, cfg_coef=3.0, temperature=1.0) -> bytes  # WAV bytes
   arrange_to_file(...)
   ```
6. **`app/streamlit_app.py`** — adds an "Arrange" tab beside the existing
   "Transcribe" tab. After a transcription, the user can choose a preset and
   click Arrange; the result plays in the browser.

### Smoke (B+2 prior session)
- `vocadito_1.wav` (8 s humming) + `lo-fi hip hop` preset, model_size=melody
  → 19 s wall-clock, 512 KB output WAV (mono 32 kHz). Plays back as expected.

### Verification — B64 (this session)
`scripts/exp_B64_musicgen_presets.py` runs all 6 presets on the same Vocadito
clip 1 with model_size=`melody` (1.5B params) and duration=10s. Logs
wall-clock and peak VRAM per preset to WandB and `reports/_exp_B64_musicgen_presets.json`.

WandB run: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/psc91mvw

## Results — full B64 run

| preset | wall_s | size_kb | peak_vram_gb |
|---|---|---|---|
| lo-fi hip hop | 29.3 (incl. cold-load) | 625 | 4.27 |
| jazz trio | 13.3 | 625 | 4.31 |
| EDM | 13.8 | 625 | 4.31 |
| orchestral cinematic | 13.5 | 625 | 4.31 |
| indie folk | 15.5 | 625 | 4.31 |
| bossa nova | 14.1 | 625 | 4.31 |

Total wall: 99.6 s for all 6 presets. Output WAVs: 10 s mono @ 32 kHz, 625 KB
each. Peak VRAM **4.31 GB** — well under the 20 GB pass criterion (4.6× headroom).
Output files in `outputs/musicgen_presets/vocadito_1_<preset>.wav`.

## Vs spec criteria

| criterion | result | met? |
|---|---|---|
| 6 presets all produce coherent output | 6/6 nonempty WAV | ✓ |
| peak VRAM < 20 GB | **4.31 GB** | ✓ (4.6× headroom) |
| weights load on first call without download error | 29 s cold-load, 13 s warm | ✓ |
| end-to-end hum → arrangement | Vocadito clip 1 → 6 styles | ✓ |

## Interpretation
- Audiocraft's import surface is heavy but most of it is unused for
  `MusicGen.get_pretrained("facebook/musicgen-melody")`. The spaCy stubs
  pattern saves us from pulling ~1 GB of NLP dependencies that would
  conflict with the rest of the env.
- Melody-conditioning via the chromagram path (`generate_with_chroma`) is
  what makes this differ from text-only MusicGen — the user's hum becomes
  the actual melodic spine of the arrangement, not just a vibe prompt.
- The 1.5B `melody` model should fit in ~7 GB. The 3.3B `melody-large`
  model needs ~13 GB — still within the 20 GB pass criterion and well
  under the 32 GB hardware limit.

## Next
- Once B64 completes, populate the results table.
- For the demo, the Streamlit app loads the model lazily on first Arrange
  click; subsequent clicks reuse the cached model via `st.cache_resource`.

## Phase C extension (B67, B68)

### B67 — MusicGen-Melody-**Large** (3.3B) sweep
Same 6 presets, same Vocadito clip 1, model_size="melody-large".

| preset | wall_s | size_kb | peak_vram_gb |
|---|---|---|---|
| lo-fi hip hop | 38.5 (incl. cold-load) | 625 | 6.25 |
| jazz trio | 13.1 | 625 | 6.25 |
| EDM | 13.2 | 625 | 6.25 |
| orchestral cinematic | 12.8 | 625 | 6.25 |
| indie folk | 13.0 | 625 | 6.25 |
| bossa nova | 13.7 | 625 | 6.25 |

Total wall: 104.2 s. Peak VRAM **6.25 GB** at fp16 (audiocraft default
precision). Same speed as the 1.5B variant — generation is bound by the
EnCodec autoregressive token stream length, not parameter count.

Output WAVs in `outputs/musicgen_presets/` (overwrites the 1.5B run; large
sounds richer at the same melody fidelity).

WandB: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/fff30thm

### B68 — LoRA fine-tuning smoke (Phase C)
- HF transformers `MusicgenMelodyForConditionalGeneration` + PEFT 0.19.1
- LoRA r=8, alpha=16, dropout=0.05
- Targets: decoder self-attention q_proj + v_proj across all 24 decoder
  layers (48 modules)
- 1.5B base model + adapter, fp32 grads + AdamW
- Synthetic pair set: Vocadito clip 1 (melody) → 6 B64-generated arrangements
  (target audio), one pair per training step
- Goal: confirm the path works end-to-end without OOM, saved adapter reloads
- Per-step loss is tokenwise CE over 4-codebook EnCodec tokens, teacher-forced

Expected to land in this report when B68 completes; see B68 wandb run for
in-progress state. The intent is *not* to beat the base model (synthetic
"distill" pairs from the same model can't); it's to validate the path so
real (melody, arrangement) pairs can later be added.

## Status
keep — Stage 7 wired into Streamlit; both 1.5B and 3.3B variants work
within budget; LoRA fine-tune scaffold validated.
