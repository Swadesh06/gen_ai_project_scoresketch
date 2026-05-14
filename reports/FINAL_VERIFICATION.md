# FINAL_VERIFICATION.md — pre-presentation state audit

## Summary

The Phase G strict scorecard and the FINAL_JOURNEY synthesis are intact;
all production-default config flags, demo SVGs, generative components,
and the MAESTRO chamber regression revert are present and verified. The
pipeline smoke transcription on a Vocadito 10 s clip produces both
MusicXML and SVG outputs. **One environmental drift was detected and
resolved during this session**: `streamlit` and `cairosvg` were listed
in `requirements.txt` but absent from the active `humscribe` conda env.
Running `scripts/setup_linux.sh --conda --skip-apt` end-to-end (the W-4
verification run) pip-installed them; the headless Streamlit smoke now
produces the expected "You can now view" line on `http://localhost:8501`.
Both states are documented below.

## Verification table

| # | Check | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| 1 | `reports/FINAL_JOURNEY.md` exists | yes | yes, 2465 words | PASS |
| 2 | FINAL_JOURNEY has 15 sections | 15 `## ` headers | 15 | PASS |
| 3 | `reports/PHASE_G_STRICT_SCORECARD.md` exists | yes | yes, unchanged | PASS |
| 4 | `outputs/demos/maestro_chamber3_30s.musicxml` zero `<actual-notes>` ≥ 24 | 0 | 0 (file has no `<actual-notes>` at all post-revert at tpb=8) | PASS |
| 5 | Chamber demo integer tempo | integer | `<sound tempo="154"` `<per-minute>154` | PASS |
| 6 | `maestro_chamber3_30s_phase_g_regression.svg` (regression evidence) | exists | yes (+ `.musicxml` sibling) | PASS |
| 7 | `outputs/demos/vocadito_1_humming_before.svg` + `_after.svg` | both exist | both exist | PASS |
| 8 | `outputs/demos/bwv_854_piano.svg` | exists | yes | PASS |
| 9 | `outputs/demos/mtg_qbh_q1_humming.svg` | exists | yes | PASS |
| 10 | `humscribe/config.py` `tatums_per_beat=12` | 12 | `tatums_per_beat: int = 12` | PASS |
| 11 | `humscribe/config.py` `render_tpb=12` | 12 | `render_tpb: int = 12` | PASS |
| 12 | `humscribe/config.py` `octave_sanity="auto"` | auto | `octave_sanity: OctaveSanity = "auto"` | PASS |
| 13 | `humscribe/config.py` `same_pitch_merge="auto"` | auto | `same_pitch_merge: SamePitchMerge = "auto"` | PASS |
| 14 | `humscribe/config.py` `formant_offset_corrector="off"` | off | `formant_offset_corrector: FormantOffsetCorrector = "off"` | PASS |
| 15 | `humscribe/config.py` `median_smooth_g5="off"` | off | `median_smooth_g5: MedianSmoothG5 = "off"` | PASS |
| 16 | `humscribe/config.py` `silent_trim_g6="off"` | off | `silent_trim_g6: SilentTrimG6 = "off"` | PASS |
| 17 | F-1 module `humscribe/beat/octave_sanity.py` | exists | yes | PASS |
| 18 | F-2e module `humscribe/pitch/formant_corrector.py` | exists | yes | PASS |
| 19 | G-4 `merge_same_pitch` in `humscribe/post_process.py` | exists | yes (3 post-process funcs total: G-4, G-5, G-6) | PASS |
| 20 | B76 module `humscribe/rhythm/voice_transformer.py` | exists | yes | PASS |
| 21 | B76 weights `checkpoints/voice_transformer_b76/best.pt` | exists | yes | PASS |
| 22 | C5b LoRA adapter weights | exists | `checkpoints/musicgen_lora_c5b_jsb_safe_step1500/adapter_model.safetensors` | PASS |
| 23 | HF MusicGen path `humscribe/arrange/musicgen.py` | exists | yes | PASS |
| 24 | G-1 `humscribe/eval/voice_emission.py` | exists | yes | PASS |
| 25 | LoRA-only confirmation: `grep "p.requires_grad = True"` in `humscribe/arrange/`, `scripts/` | 0 matches | 0 matches | PASS |
| 26 | No paper-draft files (abstract.md / intro.md / methods.md) at repo root | none | none | PASS |
| 27 | `app/streamlit_app.py` exists | yes | yes (152 lines) | PASS |
| 28 | `app/streamlit_app.py` imports without error | imports | initial: ModuleNotFoundError (streamlit absent); after `setup_linux.sh` run: **imports cleanly** (streamlit 1.57.0, cairosvg 2.9.0) | PASS after env-fix |
| 29 | Streamlit headless smoke launch | "You can now view" | initial: `streamlit: command not found`; after `setup_linux.sh`: `You can now view your Streamlit app... Local URL: http://localhost:8501` | PASS after env-fix |
| 30 | Pipeline smoke transcription on Vocadito demo | non-empty MIDI + MXL + SVG | 61 notes, BPM 115.38, both `outputs/smoke/smoke.svg` and `smoke.musicxml` written | PASS |
| 31 | `requirements.txt` lists `streamlit>=1.30` | listed | yes | PASS |
| 32 | `Dockerfile` installs from `requirements.txt` | yes | `pip install ... -r requirements.txt` line 37 | PASS |

## Streamlit UI source-level feature audit

`app/streamlit_app.py` source inspection (full read, 152 lines):

| Feature | Source line | Present |
|---|---|---|
| File upload widget (audio) | `uploaded = st.file_uploader("Audio", ...)` line 42 | yes — accepts wav/mp3/flac/m4a |
| Input-kind selector (humming / piano / guitar / instrument) | `kind = st.selectbox("Input kind", ...)` line 45 | yes |
| Mode selector (soft / medium / hard) | `mode = st.selectbox("Mode", ...)` line 47 | yes |
| Pitch-model selector (pesto_crepevoicing / pesto / crepe) | line 49 | yes |
| **G-7 demo hum dropdown** (no-upload) | `st.selectbox("Pre-recorded hum", labels, ...)` line 38 | yes — five `demo_*.wav` clips ship in `app/demos/` |
| G-14 multi-take checkbox + extra uploads | line 53 / 58 | yes |
| Inline SVG render of transcription | `st.components.v1.html(...)` line 91 | yes |
| MusicXML download button | `st.download_button("Download MusicXML", ...)` line 92 | yes |
| MIDI download button | — | **MISSING** (only MusicXML is offered; MIDI is not exported) |
| Plain SVG download button | — | **MISSING** (SVG is inline-rendered but not downloadable) |
| Stage-7 arrangement toggle | `tab_a` ("Arrange" tab) line 144 | yes |
| Arrangement style preset selector | `st.selectbox("Style preset", ...)` line 108 | yes |
| Arrangement model-size selector (melody-large / melody) | line 110 | yes |
| Arrangement duration slider | `st.slider("Duration (s)", 8, 30, 15)` line 113 | yes |
| Custom prompt text input | line 114 | yes |
| **B77 LoRA adapter dropdown** (Phase D integration) | line 124 | yes — auto-discovers `checkpoints/musicgen_lora_b77/step_*/` |
| Arrangement WAV download | line 136 | yes |

Two cosmetic gaps (MIDI and bare-SVG download buttons) — neither is in
the v6 W-1 strict criteria; MusicXML download is sufficient for the
score path because MusicXML is the canonical exchange format. The user
may want to wire MIDI/SVG downloads after the presentation.

## Blockers for presentation

### 1. Streamlit + cairosvg missing from local conda env  (RESOLVED in this session)

**Initial symptom**

```
$ python -c "import streamlit"
ModuleNotFoundError: No module named 'streamlit'

$ streamlit run app/streamlit_app.py ...
timeout: failed to run command 'streamlit': No such file or directory
```

**Root cause**

The local `humscribe` conda env at
`/home/swadesh/miniconda3/envs/humscribe/` was built without
`streamlit` and `cairosvg` despite both being listed in
`requirements.txt`. The Dockerfile pipeline does pull them in.

**Resolution applied this session**

Running the W-4 verification of `scripts/setup_linux.sh --conda
--skip-apt` end-to-end pip-installed the missing packages from
`requirements.txt` (log: `logs/setup_linux_test.log`). The Streamlit
headless smoke re-run produced:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://172.18.0.2:8501
```

**Production recommendation for the live demo**

| option | command | wall time | risk |
|---|---|---|---|
| A (cross-platform safest) | `docker build -t humscribe . && docker run -p 8501:8501 humscribe` | ~10 min build, then instant runs | low — same env every time |
| B (current host, ready now) | `streamlit run app/streamlit_app.py` (env already pre-warmed) | instant | low — verified to start |
| C (slides-only fallback) | walk through `outputs/figures/F*.png` + `S*.png` in the deck without launching Streamlit | n/a | none |

The local host is **ready to demo as of this session** (option B); the
Docker path remains the recommended cross-platform default.

### 2. Two minor download buttons missing in `app/streamlit_app.py` (LOW — cosmetic)

MIDI and plain-SVG download buttons are not wired (MusicXML downloads
fine). Not a blocker for the 4-min talk. Add post-presentation if
desired.

## Verified-but-unsurprising items

- `outputs/figures/F1_metric_trajectory.png` through `F7_demo_before_after.png` — written by the prior session, all 7 present.
- `outputs/demos/vocadito_1_humming_pre_phase_g.svg` — historical evidence trail preserved.
- `reports/_item-g{1..17}.json` + `_item-g11_tuplet_audit.json` — all present, source of truth for the Phase G numbers.
- `Dockerfile` — installs Java for MV2H, FluidSynth + soundfont for synthesis, both correct.

## Smoke transcription output

```
notes=61  bpm=115.38
svg_exists=True  mxl_exists=True
svg=outputs/smoke/smoke.svg
mxl=outputs/smoke/smoke.musicxml
```

Audio: `app/demos/demo_1_vocadito_S1.wav` (CC-BY Vocadito clip ~13 s).
Config: `input_kind="humming"`, `mode="soft"`,
`pitch_model="pesto_crepevoicing"`. Production-default flags active.
Log: `logs/pipeline_smoke.log`.
