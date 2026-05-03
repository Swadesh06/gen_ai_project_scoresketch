# HumScribe v3.4

Audio → MusicXML/SVG score, with optional generative arrangement.

The pipeline is built around frozen pretrained models: monophonic input
(humming, voice) goes through pitch tracking + voicing segmentation;
polyphonic input (piano, guitar, instruments) goes through a transformer
transcriber. Both branches share the same beat tracker, rhythm-quantization
DP, music21 score build, and Verovio SVG renderer. An optional Stage 7
takes the transcribed melody back into MusicGen-Melody to produce a styled
arrangement.

## Pipeline

```
                                  Stage 6
                                  music21 + Verovio
                                  → MusicXML / SVG
                                  ↑
audio →  Stage 1   Stage 2A      Stage 4     Stage 5
         load       PESTO + CREPE  beat_this   Cemgil-Kappen
         resample   pitch + voice  beats+bpm   DP + voice tracking
                    (humming)         |          ↑
                                      |         tatums (24/beat)
                    Stage 2B          |
                    YourMT3+ T5       |
                    (piano default)   |
                    ByteDance opt-in  |
                    basic_pitch opt   |
                                      ↓
                                Stage 7 (optional)
                                MusicGen-Melody
                                + 6 style presets
                                → arrangement WAV
```

- **Stage 1** — `humscribe.audio_io`. Soundfile-based load + resample.
- **Stage 2A** — `humscribe.pitch.{pesto_track, crepe_track, ensemble}`.
  Default for humming: **PESTO pitch + CREPE periodicity-as-voicing** hybrid
  (B36/B36b: +5pp Vocadito).
- **Stage 2B** — `humscribe.instrument.{yourmt3plus, piano, basic_pitch}`.
  Default for piano: **YourMT3+** (Chang et al. 2024; B63: +6.1pp 9-piece
  ASAP mean over ByteDance). Pass `transcriber="bytedance_piano"` for the
  older fast path; `basic_pitch` for non-piano polyphony.
- **Stage 4** — `humscribe.beat.beat_this_track`. beat_this from
  Heidelberg, with a tempo-octave correction (B13: +6pp Stage-4).
- **Stage 5** — `humscribe.rhythm.{viterbi_quantize, voice_tracking, voice_transformer}`.
  Cemgil-Kappen DP at 24 tatums/beat for the snap metric, requantized to
  TPB=12 for SVG render. Voice tracking with adaptive pitch-jump (B49:
  +1.9pp on mixed ASAP). Phase D: optional **B76 Transformer voice tracker**
  (94.5% mean acc on Romantic ASAP, Liszt 90.8%, Beethoven 97.4%) +
  per-voice independent DP, auto-routed for melody+accompaniment pieces
  (+1.7pp on Chopin Berceuse).
- **Stage 6** — `humscribe.score`. music21 stream build with
  Krumhansl–Schmuckler key estimation (B+2 item 1.4); Verovio SVG render.
- **Stage 7** *(optional)* — `humscribe.arrange.musicgen`. MusicGen-Melody
  1.5B (or 3.3B `melody-large`, the new Streamlit default) with 6 prompt
  presets. Peak 4.3 GB VRAM (1.5B) or 6.25 GB (3.3B). Phase D: optional
  **PEFT LoRA adapter** (B77, 0.34% trainable params, 69% loss decay) —
  pass `lora_adapter="checkpoints/musicgen_lora_b77/step_300"` for
  fine-tuned style/speaker behavior.

## Headline metrics (B+2)

| metric | result | vs Phase A |
|---|---|---|
| MIR-1K mean RPA (5 clips) | 0.988 | unchanged |
| ASAP BWV 846 beat-F | 0.915 | unchanged |
| ASAP BWV 846 Stage-5 snap | **0.878** (YMT3+) | **+15.4pp** |
| ASAP 5-Bach Fugue mean snap | **0.898** (YMT3+) | **+12.5pp** |
| ASAP Beethoven Sonata 21-1 snap | **0.897** (YMT3+) | first-class |
| ASAP Schumann Toccata snap | **0.846** (YMT3+) | first-class |
| ASAP Chopin Berceuse snap | **0.675** (YMT3+) | first-class |
| ASAP 9-piece overall mean snap | **0.774** (YMT3+) | first-class |
| Vocadito A1 soft no-offset F1 (40 clips) | **0.665** | **+12.7pp** |
| Vocadito A2 soft no-offset F1 | **0.630** | +10.5pp |
| Vocadito IAA ceiling (no-offset) | 0.740 | reference only |
| MAESTRO instrument F1 (5-piece sanity) | 0.984 | first-class |
| MTG-QBH visual nonempty (10 clips) | 100% | unchanged |
| **B76 voice tracker on held-out Romantic ASAP** | **94.47% mean** | first-class (Phase D) |
| ↳ Liszt Sonata voice acc | **90.8%** | first-class |
| ↳ Beethoven 21-1 voice acc | **97.4%** | first-class |
| **B77 LoRA fine-tune loss decay** (300 steps, real chroma) | **69%** | first-class (Phase D) |
| **B79/B80 per-voice DP wins on Chopin Berceuse snap-F1** | **+1.66pp** | first-class (Phase D) |

## Repo layout

```
humscribe/        — package code (importable as humscribe.*)
  pitch/          — PESTO, CREPE, hybrid voicing
  beat/           — beat_this wrapper
  rhythm/         — Cemgil-Kappen DP + voice tracking
  instrument/     — YourMT3+, ByteDance piano, basic_pitch
  arrange/        — MusicGen-Melody (Stage 7)
  datasets/       — mtg_qbh loader (mirdata 1.0.0 doesn't ship one)
  config.py       — PipelineConfig + ModeConfig
  pipeline.py     — top-level transcribe()
  score.py        — music21 + Verovio
  audio_io.py     — load/resample
  notes.py        — NoteEvent dataclass
app/              — Streamlit UI (Transcribe + Arrange tabs)
scripts/          — gates, exp_B*, sweeps
reports/          — per-experiment markdown + JSON; INDEX.md is the table
task_descriptions/— spec docs
checkpoints/      — gitignored
logs/             — gitignored (WandB is source of truth)
outputs/          — gitignored (generated SVGs / MusicXML / arrangement WAVs)
```

## Quickstart

```bash
conda activate humscribe
cd /workspace/swadesh/gen_ai_project_scoresketch
set -a && source .env && set +a

# Streamlit (Transcribe + Arrange tabs)
streamlit run app/streamlit_app.py

# CLI: transcribe a humming clip
python -c "
from humscribe.pipeline import transcribe
from humscribe.config import PipelineConfig
r = transcribe('vocadito_1.wav',
               PipelineConfig(input_kind='humming', mode='soft',
                              pitch_model='pesto_crepevoicing',
                              musicxml_path='out.xml', svg_path='out.svg'))
print(r.n_notes, 'notes,', r.bpm, 'bpm')
"

# CLI: transcribe a piano recording (defaults to YourMT3+)
python -c "
from humscribe.pipeline import transcribe
from humscribe.config import PipelineConfig
r = transcribe('bach.wav',
               PipelineConfig(input_kind='piano',
                              musicxml_path='out.xml', svg_path='out.svg'))
print(r.n_notes, 'notes')
"

# Arrange (Stage 7): hum → arrangement
python -c "
from humscribe.arrange.musicgen import arrange_to_file
arrange_to_file('vocadito_1.wav', 'jazz trio with brushed drums',
                'out.wav', duration_s=10.0, model_size='melody')
"

# Arrange with a fine-tuned LoRA adapter (B77)
python -c "
from humscribe.arrange.musicgen import arrange_to_file
arrange_to_file('vocadito_1.wav', 'jazz trio with brushed drums',
                'out.wav', duration_s=15.0, model_size='melody',
                lora_adapter='checkpoints/musicgen_lora_b77/step_300')
"

# Force per-voice DP + B76 transformer voice tracker (Phase D, Chopin-style pieces)
python -c "
from humscribe.pipeline import transcribe
from humscribe.config import PipelineConfig
r = transcribe('chopin_berceuse.wav',
               PipelineConfig(input_kind='piano', per_voice_dp='on'))
print(r.n_notes, 'notes')
"
```

## Reproducing the headline numbers

```bash
# Vocadito (humming):
python scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing \
       --mode soft --annotator A1
# → mean F1 = 0.665

# ASAP single-piece (BWV 846, ByteDance baseline):
python scripts/gate_asap_rhythm.py
# → snap = 0.847

# ASAP 5-Bach mean (ByteDance):
python scripts/exp_B12_asap_multi.py --n-pieces 5
# → mean snap = 0.859

# ASAP 9-piece mean (BD vs YMT3+):
python scripts/exp_B63_yourmt3_asap.py
# → BD mean 0.713 / YMT3+ mean 0.774

# MAESTRO instrument sanity:
python scripts/exp_B14_maestro_instrument.py --n-pieces 5
# → mean F1 = 0.984

# All 6 MusicGen presets:
python scripts/exp_B64_musicgen_presets.py
# → all 6 nonempty, peak 4.31 GB
```

## License matrix

| component | license | url |
|---|---|---|
| pipeline code (this repo) | MIT (private until release) | — |
| PESTO pitch tracker | MIT | github.com/SonyCSLParis/pesto |
| CREPE pitch tracker | MIT | github.com/marl/crepe |
| beat_this beat tracker | CC-BY-NC-SA-4.0 | github.com/CPJKU/beat_this |
| YourMT3+ transcriber (default for piano) | Apache-2.0 | github.com/mimbres/YourMT3 |
| ByteDance piano transcription (opt-in) | Apache-2.0 | github.com/bytedance/piano_transcription |
| basic_pitch (opt-in) | Apache-2.0 | github.com/spotify/basic-pitch |
| audiocraft / MusicGen code | MIT | github.com/facebookresearch/audiocraft |
| MusicGen-Melody weights | **CC-BY-NC-4.0** | huggingface.co/facebook/musicgen-melody |
| music21 | BSD-3-Clause | github.com/cuthbertLab/music21 |
| Verovio | LGPL-3.0 | github.com/rism-digital/verovio |
| MIR-1K dataset | research-use | mirlab.org |
| ASAP dataset | CC-BY-NC-4.0 | github.com/CPJKU/asap-dataset |
| Vocadito | CC-BY-4.0 | mtg.upf.edu/vocadito |
| MTG-QBH | CC-BY-NC-SA-4.0 | mtg.upf.edu/mtg-qbh |

The arrangement weights (MusicGen-Melody) are CC-BY-NC-4.0 → Stage 7 outputs
are non-commercial use only. The transcription path (Stages 1-6) does not
inherit this restriction.

## Citations

- **PESTO** — Riou et al., "PESTO: Pitch Estimation with Self-supervised
  Transposition-equivariant Objective" (ISMIR 2023).
  [arXiv:2309.02265](https://arxiv.org/abs/2309.02265)
- **CREPE** — Kim et al., "CREPE: A Convolutional Representation for
  Pitch Estimation" (ICASSP 2018).
  [arXiv:1802.06182](https://arxiv.org/abs/1802.06182)
- **beat_this** — Foscarin et al., "Beat This! Accurate Beat Tracking
  Without DBN Postprocessing" (ISMIR 2024).
  [arXiv:2407.21658](https://arxiv.org/abs/2407.21658)
- **YourMT3+** — Chang et al., "YourMT3+: Multi-Instrument Music
  Transcription with Mixture-of-Experts T5" (MLSP 2024).
  [arXiv:2407.04822](https://arxiv.org/abs/2407.04822)
- **ByteDance piano** — Kong et al., "High-Resolution Piano Transcription
  with Pedals by Regressing Onset and Offset Times" (TASLP 2021).
  [arXiv:2010.01815](https://arxiv.org/abs/2010.01815)
- **basic_pitch** — Bittner et al., "A Lightweight Instrument-Agnostic
  Model for Polyphonic Note Transcription" (ICASSP 2022). Spotify Research.
- **MusicGen** — Copet et al., "Simple and Controllable Music Generation"
  (NeurIPS 2023). [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)
- **Cemgil-Kappen DP** — Cemgil & Kappen, "Monte Carlo methods for tempo
  tracking and rhythm quantization" (JAIR 2003).
- **Krumhansl-Schmuckler** — Krumhansl, *Cognitive Foundations of Musical
  Pitch* (Oxford, 1990); Krumhansl & Schmuckler key-finding algorithm.

## Status

Phase B+2 stable + **Phase D extensions integrated**. All four spec gates pass.
100+ commits, 100+ WandB runs at
[wandb.ai/agam_p-iit-roorkee/humscribe-v3.2](https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2).
Per-experiment reports under `reports/exp_B*.md` and `reports/item-*.md`;
the index of all experiments is at `reports/INDEX.md`.

Phase D summary: `reports/PHASE_D_SUMMARY.md`. Integration details:
`reports/PHASE_D_INTEGRATION.md`.

For the live plan see `PLAN.md`. For the agent's operating contract see
`CLAUDE.md`. For the next-steps spec see
`task_descriptions/task_description_v2.md`.
