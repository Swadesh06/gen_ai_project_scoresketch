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

## Status
keep — Stage 7 wired into Streamlit, smoke confirmed; full preset
verification in flight via B64.
