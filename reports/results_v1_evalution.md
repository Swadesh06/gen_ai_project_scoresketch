# HumScribe v3.4 — next-steps with generative AI integrations

> **Hardware**: RTX Pro 4500 Blackwell, **32 GB VRAM**, always-on. No session caps.
>
> **What this is**: A consolidated next-steps spec that combines (a) the polish work flagged in the result evaluation, (b) the two generative-AI integrations (YourMT3+ for transcription, MusicGen-Melody for arrangement), and (c) the remaining items from the result evaluation. Architecture for the existing six stages is unchanged; two new components are added (a generative transcriber as a Stage 2-A alternative, and a generative arrangement post-stage as Stage 7).
>
> **Apply on top of the agent's Phase B+1 codebase.**
>
> **No time estimates anywhere in this document** — just deliverables, success criteria, and dependencies. Do the work as fast as it can correctly be done.

---

## What changes vs the previous plan

| Aspect | Before | Now |
|---|---|---|
| Compute | 60 GPU-hr cap (Colab/Kaggle), 12-hr session caps | Always-on 32 GB Blackwell |
| Largest model | ByteDance piano (~330 MB) | **MusicGen-Melody-Large (3.3B, ~13 GB VRAM)** |
| Generative components | None real | **2 genuine generative AI components** |
| Demo scope | Audio → score | Audio → score → **arranged multi-instrument output** |
| Course-AI fit | Weak (mostly discriminative MIR) | **Strong (seq2seq generation + autoregressive audio generation)** |

---

## VRAM budget on the Blackwell

Sanity check that everything fits — runs are sequential, not concurrent, so you only need to fit the largest single model at a time:

| Component | VRAM | When |
|---|---|---|
| ByteDance piano | ~3 GB | always (default piano backend) |
| Basic Pitch | <1 GB | guitar/multi |
| **YourMT3+** | **~5 GB** | new: piano alternative for Romantic |
| PESTO | <1 GB | always (humming pitch) |
| CREPE-full | ~2 GB | always (humming voicing) |
| beat_this | ~2 GB | always |
| **MusicGen-Melody-Large** | **~13 GB at fp16** | new: optional arrangement post-stage |
| Headroom | ~14 GB | for activations, cache, batch experiments |

You can run the full pipeline including arrangement in <20 GB peak. The 32 GB Blackwell is the right size; 16 GB would have been tight on MusicGen-Large.

---

## Work item 1 — Demo-critical rendering fixes

**Why first**: from the evaluation, the agent's MAESTRO score sounds at F1=0.984 but the rendered SVG has 12-lets, 24-lets, and 48-lets that no human would ever read. Fix this before showing the system to anyone. Zero risk to numbers.

### 1.1. Round tempo display to integer

In `humscribe/score.py` (or wherever the music21 Score is built):

```python
score.insert(0, music21.tempo.MetronomeMark(number=int(round(bpm))))
```

Replaces `♩ = 73.17073170731705` with `♩ = 73`.

### 1.2. Cap tuplet denominators in the DP lattice

In `humscribe/rhythm/viterbi_quantize.py`, after building the candidate state set per note and before Viterbi:

```python
from fractions import Fraction
ALLOWED_DENOMS = {1, 2, 3, 4, 6, 8, 12, 16}  # what humans actually notate

def _denom_ok(s, tpb):
    return Fraction(s, tpb).limit_denominator(16).denominator in ALLOWED_DENOMS

candidates = [s for s in candidates if _denom_ok(s, tatums_per_beat)]
```

This prunes positions like `7/24` of a beat (which produce 24-lets) before they enter the lattice.

**Pass criterion**: re-run `gate_asap_rhythm.py` and `exp_B12_asap_multi.py`. Snap metric should drop ≤1pp on Bach Fugues. If it drops more than that, raise `ALLOWED_DENOMS` ceiling to 24 and retest.

### 1.3. Render with TPB=12, score with TPB=24

Keep TPB=24 internally for measuring snap accuracy (the agent's optimization target). When constructing the music21 Score for rendering, requantize to TPB=12. Two-line change at the score-build boundary.

**Pass criterion**: Bach BWV 854 SVG should have triplets and sextuplets only — no 12-lets, no 24-lets, no 48-lets. Visual diff vs the current `bwv_854_piano.svg` confirms.

### 1.4. KrumhanslSchmuckler key estimation

Run once on the produced MIDI before passing to music21:

```python
from music21.analysis.discrete import KrumhanslSchmuckler
key = KrumhanslSchmuckler().getSolution(stream)
score.insert(0, key)
```

Collapses explicit accidentals (every `D#` becomes implicit when the key is E major). Cleans up notation by ~30%.

### 1.5. Side-by-side rendered SVG in every future experiment template

Add `scripts/compare_svgs.py` that renders `before.svg` and `after.svg` next to each other for any A/B run. The agent's gate template only measured numeric metrics; adding visual diff would have caught the over-complex tuplet issue automatically.

**Pass criterion (overall for work item 1)**: re-run all three demo files (`bwv_854_piano.svg`, `maestro_chamber3_30s.svg`, `mtg_qbh_q1_humming.svg`). Each should have integer BPM, no tuplets above 6, a key signature, and no regression on snap metric.

---

## Work item 2 — YourMT3+ as a generative seq2seq transcription backend

**Why this matters**: B58 in the agent's run did the most important diagnostic of the whole project — using oracle inputs to prove that **100% of remaining ASAP loss is in ByteDance**, with beat tracking and DP both essentially perfect. Headroom: +18.8pp on Romantic ASAP if the transcriber is fixed. YourMT3+ is the natural drop-in.

**Why this is generative AI**: YourMT3+ is a T5 encoder-decoder transformer that **autoregressively generates a sequence of MIDI tokens** from a mel-spectrogram, one token at a time, exactly like a language model emitting words.

**Why YourMT3+ specifically and not vanilla MT3**: MT3 (Gardner et al. 2022) is the original, but its checkpoint is in T5X format and needs conversion. YourMT3+ (Chang et al. MLSP 2024, `https://arxiv.org/abs/2407.04822`) is the reproducible follow-up, available on HuggingFace, with a free-GPU Spaces demo, Colab notebook, and pre-release inference code. Apache-2.0. Trained on a much broader stem-augmented dataset than MAESTRO so it generalizes better to Romantic music.

### 2.1. Architecture position

YourMT3+ slots in alongside ByteDance and Basic Pitch as a third option:

```
Stage 2-A (instrument transcription) backends:
- bytedance     → CRNN, MAESTRO-saturated, fast (default)
- basic_pitch   → small CNN, guitar-best, fast
- yourmt3plus   → T5 seq2seq, broadest generalization, slower (NEW)
```

Extend the agent's existing `auto_piano` heuristic (which currently switches to `basic_pitch` for slow chordal pieces): detect Romantic-style content (median IOI > 0.6 s + dense pedaled chords) and route to YourMT3+. ByteDance keeps the default for MAESTRO-style classical (where it dominates).

### 2.2. Concrete integration

Add `humscribe/instrument/yourmt3plus.py`:

```python
"""YourMT3+ piano transcription backend.
Source: https://github.com/mimbres/YourMT3 (MLSP 2024)
License: Apache-2.0
~5 GB VRAM at fp16 inference.
"""
from pathlib import Path
import torch
from humscribe.notes.note_event import NoteEvent

class YourMT3Plus:
    def __init__(self, device="cuda", dtype=torch.float16):
        from yourmt3p import load_pretrained  # from pip install or git clone
        self.model = load_pretrained(
            "yourmt3p_pop_uvae", device=device, dtype=dtype,
        )
        self.device = device

    @torch.inference_mode()
    def transcribe(self, audio_path: str) -> list[NoteEvent]:
        midi = self.model.predict(audio_path)
        return [
            NoteEvent(onset_s=n.start, offset_s=n.end,
                      midi_pitch=n.pitch, velocity=n.velocity)
            for inst in midi.instruments for n in inst.notes
        ]
```

Wire into the dispatch in `humscribe/pipeline.py` next to existing backends, and update `auto_piano` to route Romantic-detected pieces to YourMT3+ instead of basic_pitch.

### 2.3. Pass criteria — verify before keeping it as default

Run all four ASAP test sets before/after. Targets, derived from B58's oracle ceiling:

| Test | ByteDance current | YourMT3+ target | Source |
|---|---|---|---|
| MAESTRO instrument F1 | 0.984 | ≥ 0.95 (no regression) | sanity |
| ASAP Bach 5-Fugue mean snap | 0.856 | ≥ 0.84 (within noise) | preserve win |
| ASAP 5-mixed mean snap | 0.590 | **≥ 0.74** | this is the upside |
| Liszt Sonata snap | 0.078 | ≥ 0.20 | structural unsalvageable |
| Beethoven Sonata 21-1 snap | 0.811 | **≥ 0.92** | B58 oracle was 0.982 |
| Schumann Toccata snap | 0.745 | **≥ 0.93** | B58 oracle was 0.975 |

The Beethoven and Schumann lifts are where YourMT3+ will pay back integration effort. Liszt is structurally DP-bound (oracle was 0.132) — don't expect miracles there.

**Decision rule**: if Beethoven snap ≥ 0.85 AND mixed mean ≥ 0.70, keep YourMT3+ as default for Romantic-detected pieces. If both targets miss, leave it behind a `--piano-backend yourmt3plus` flag for ablation only.

---

## Work item 3 — MusicGen-Melody for arrangement (new Stage 7)

**Why this matters**: This is the demo flourish that converts the project from "MIR pipeline" to "generative AI for music". The user hums, the system transcribes to MIDI, then MusicGen-Melody generates a full multi-instrument arrangement (drums, bass, accompaniment) in a chosen style — using the user's hum as the *melodic* conditioning, not just text. Meta's MusicGen-Melody-Large was specifically designed for this: it accepts a melody waveform as conditioning input and generates 32 kHz audio that follows the melody.

**Why this is generative AI**: autoregressive Transformer generating discrete EnCodec audio tokens. Same architecture family as language models, applied to audio. Unambiguously generative.

### 3.1. Architecture position

This is a new **Stage 7**, optional, runs after Stage 6 produces the MIDI:

```
Stage 6: MusicXML / MIDI / SVG (existing)
          │
          ▼
Stage 7: Arrangement (NEW, optional)
   Input:  user's original audio (humming) + transcribed MIDI + style prompt
   Model:  facebook/musicgen-melody-large (3.3B)
   Output: 32 kHz stereo audio of full arrangement
```

The Streamlit UI gets a new tab "Arrange" with:
- Style preset dropdown (text prompts like "lo-fi hip hop", "jazz trio", "EDM", "orchestral cinematic", "indie folk", "bossa nova")
- Free-text prompt override
- Duration slider (8 / 15 / 30 s)
- "Generate arrangement" button
- Resulting `st.audio` widget for playback + download

### 3.2. Why MusicGen-Melody specifically

| Variant | Size | What it does | Fit for HumScribe |
|---|---|---|---|
| musicgen-small | 300M | text → music | ⚠️ ignores melody |
| musicgen-medium | 1.5B | text → music | ⚠️ ignores melody |
| musicgen-large | 3.3B | text → music | ⚠️ ignores melody |
| **musicgen-melody-large** | **3.3B** | **text + melody → music** | ✅ **uses your hum as the melody** |
| musicgen-stereo-large | 3.3B | stereo text → music | Stereo, but no melody |
| musicgen-melody | 1.5B | text + melody → music | Lighter melody variant — fallback if VRAM tight |

Melody-large is the right choice — it accepts a chromagram of your input audio and arranges around it. Without melody conditioning, the model would just generate something matching the text prompt and ignore what the user actually hummed.

### 3.3. Concrete integration

Add `humscribe/arrange/musicgen.py`:

```python
"""MusicGen-Melody arrangement post-stage.
Source: https://huggingface.co/facebook/musicgen-melody-large
License: code MIT, weights CC-BY-NC-4.0
~13 GB VRAM at fp16, ~30 s for 15 s of audio on Blackwell.
"""
import torch, torchaudio, scipy.io.wavfile

class MusicGenArranger:
    def __init__(self, model_size="melody-large", device="cuda", dtype=torch.float16):
        from audiocraft.models import MusicGen
        self.model = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}",
                                             device=device)
        # convert to fp16 for VRAM efficiency
        for p in self.model.lm.parameters(): p.data = p.data.to(dtype)

    def arrange(self, melody_audio_path: str, prompt: str,
                duration_s: float = 15.0) -> bytes:
        """Generate an arrangement conditioned on the user's audio + a style prompt."""
        melody, sr = torchaudio.load(melody_audio_path)
        if sr != 32000:
            melody = torchaudio.functional.resample(melody, sr, 32000)
        if melody.shape[0] > 1:
            melody = melody.mean(0, keepdim=True)
        melody = melody.unsqueeze(0)  # (1, 1, T)

        self.model.set_generation_params(duration=duration_s, top_k=250, top_p=0.0,
                                         temperature=1.0, cfg_coef=3.0)
        wav = self.model.generate_with_chroma(
            descriptions=[prompt], melody_wavs=melody, melody_sample_rate=32000,
        )
        out = wav[0].cpu().numpy().T
        import io
        buf = io.BytesIO()
        scipy.io.wavfile.write(buf, 32000, (out * 32767).astype("int16"))
        return buf.getvalue()
```

Streamlit hook in `app/streamlit_app.py`:

```python
with st.expander("🎼 Generate arrangement (uses your hum as the melody)"):
    style = st.selectbox("Style", [
        "lo-fi hip hop with mellow piano", "energetic EDM with synth bass",
        "jazz trio with acoustic guitar", "orchestral cinematic strings",
        "indie folk with acoustic guitar and brushed drums",
        "bossa nova with nylon guitar"
    ])
    custom_prompt = st.text_input("Custom prompt (optional)", "")
    dur = st.slider("Duration (seconds)", 8, 30, 15)
    if st.button("Arrange", type="secondary"):
        with st.spinner("Generating arrangement…"):
            from humscribe.arrange.musicgen import MusicGenArranger
            ar = MusicGenArranger()
            wav_bytes = ar.arrange(audio_input_path, custom_prompt or style, duration_s=dur)
        st.audio(wav_bytes, format="audio/wav")
        st.download_button("Download arrangement", wav_bytes, "arrangement.wav")
```

### 3.4. Resource notes

- **Disk**: ~7 GB for musicgen-melody-large weights (auto-downloaded by `audiocraft` on first call to `~/.cache/huggingface/`).
- **VRAM**: ~13 GB at fp16. Comfortably fits in 32 GB.
- **Cache**: add `@st.cache_resource` on the arranger constructor so Streamlit doesn't reload weights on every request.
- **License**: code MIT, weights CC-BY-NC-4.0. **Same constraint as PESTO** — fine for course project, document.

### 3.5. Pass criteria

- End-to-end: user hums Twinkle, transcription works, MusicGen receives the hum as melody conditioning + a style prompt, generated audio is recognizably Twinkle in the chosen style. Verified by ear, not metric.
- VRAM peak during arrangement < 20 GB.
- Weights load on first call without download errors.
- All 6 style presets produce coherent output (no garbage outputs from prompt mismatch).

### 3.6. The course-paper sentence this unlocks

You can now legitimately write:

> *"HumScribe combines discriminative components (CREPE/PESTO pitch tracking, ByteDance/Basic Pitch transcription) with generative components (YourMT3+ for sequence-to-sequence symbolic transcription on hard polyphonic music, and MusicGen-Melody for melody-conditioned audio arrangement) — the discriminative components handle bounded subproblems where they outperform generative methods, the generative components handle open-ended generation tasks where they uniquely apply."*

That is a defensible architecture story for a Generative AI course paper.

---

## Work item 4 — Voicing exit-side hysteresis on humming

**Why this matters**: agent's B55 found Vocadito no-offset F1 = 0.665 but offset-strict F1 = 0.439 — durations are off on roughly half the matched notes. The cause B55 diagnosed was vibrato-induced voicing dips ending notes early. B56 tried DP-based duration snapping and it didn't work because humming is too rubato. The actual fix is segmenter-level: once a note is "active", lower the voicing threshold required to *end* it.

The agent tried voicing hysteresis on the *entry* side in B47 (the threshold at which a note starts) and found it didn't help. They didn't try the symmetric exit-side hysteresis, which is the one that maps to the failure mode.

### 4.1. Implementation

In `humscribe/pitch/voicing.py`:

```python
def segment_with_hysteresis(voicing, vt_enter=0.75, vt_exit=0.45, min_dur_s=0.052):
    """Hysteresis: enter a note when voicing > vt_enter, stay in until voicing < vt_exit.

    Addresses B55 finding that vibrato dips end notes early.
    Default vt_exit=0.45 is well below the entry threshold; tune in 4.2.
    """
    state = "off"
    notes = []
    for i, v in enumerate(voicing):
        if state == "off" and v > vt_enter:
            state = "on"; start = i
        elif state == "on" and v < vt_exit:
            state = "off"; notes.append((start, i))
    if state == "on":
        notes.append((start, len(voicing)))
    return notes
```

Wire it in alongside the existing `segment_pitch_to_notes` function, behind a config flag `voicing_hysteresis=True` so it's a clean A/B against the current default.

### 4.2. Sweep `vt_exit`

Run a sweep `vt_exit ∈ {0.25, 0.35, 0.45, 0.55, 0.65}` with `vt_enter=0.75` fixed. Use the same `gate_vocadito_conp.py` setup the agent used for B55 — Vocadito A1 + A2, but evaluate at three offset tolerances (None, 0.5, 0.2) to make sure no-offset F1 doesn't regress.

### 4.3. Pass criteria

| Metric | Current (B55) | Target |
|---|---|---|
| Vocadito A1 no-offset F1 | 0.665 | ≥ 0.66 (no regression) |
| Vocadito A1 offset50 F1 | 0.573 | ≥ 0.60 |
| **Vocadito A1 offset20 F1** | **0.439** | **≥ 0.50** |

The IAA ceiling on offset20 F1 is 0.642, so there's room above the target — pushing for ≥ 0.55 if the sweep finds it cleanly is fine. Above 0.60 is overfitting territory.

**Decision rule**: if the best `vt_exit` configuration improves offset20 F1 by ≥ 5pp without no-offset regression, promote it as default in `ModeConfig.for_mode("soft", "pesto_crepevoicing")`.

---

## Work item 5 — Pseudo-label MedleyDB-Melody for an enlarged onset training set (speculative)

**Skip this if work items 1–4 hit deadline pressure.** Listed because the agent's BiLSTM onset detectors (B10/B19) all underperformed because of small data (40 Vocadito clips). MedleyDB-Melody has 108 vocal clips with f0 annotations.

### 5.1. The plan

1. Pseudo-label MedleyDB-Melody onsets using the existing voicing-based segmenter (the same one currently in production at F1=0.665).
2. Combine Vocadito (40, real labels) + MedleyDB-Melody (108, pseudo labels) → ~148-clip training set.
3. Re-train the B19 mel-BiLSTM onset detector on this combined set.
4. Compare against the heuristic voicing baseline.

### 5.2. Pass criterion

- Vocadito A1 no-offset F1 ≥ 0.69 (clearly above the heuristic 0.665, accounting for cross-validation noise).

If it lands below 0.66, the data is still too small or the pseudo labels are leaking heuristic errors. Discard.

### 5.3. Why speculative

The agent already tried HuBERT features in B52 and got 0.592 — well below the heuristic. The bet here is that *more data* of the right distribution would close the gap, not better features. That bet may still lose. Don't sink time into this if the demo is the priority.

---

## Work item 6 — Final demo polish (after all of the above)

### 6.1. Re-run all gates after the new components are in

Re-run, in order:

```bash
python scripts/gate_mir1k_pitch_sanity.py        # should still pass
python scripts/gate_asap_rhythm.py                # with rendering fixes from item 1
python scripts/exp_B12_asap_multi.py --n-pieces 5
python scripts/exp_B14_maestro_instrument.py --n-pieces 5
python scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing --mode soft --annotator A1
python scripts/gate_vocadito_conp.py --pitch-model pesto_crepevoicing --mode soft --annotator A2
python scripts/gate_mtg_qbh_visual.py --modes soft
```

Plus the new gates:
- ASAP mixed (5-piece) with YourMT3+ as backend
- Vocadito A1+A2 with voicing hysteresis active
- MTG-QBH with arrangement output for one clip (verify Stage 7 wires through end-to-end)

### 6.2. Report figures

Side-by-side renders:
- ScoreSketch's Twinkle (30 notes vs 14 GT) vs HumScribe's Twinkle. The headline figure of the project.
- Bach BWV 854 before rendering fixes (with 48-lets) vs after (with sane triplets). Demonstrates work item 1 was real.
- Vocadito A1 cumulative-trajectory plot from the agent's PHASE_B_FINAL.md (0.538 → 0.665).
- Optional: ASAP Beethoven snap before YourMT3+ vs after. Demonstrates work item 2 was real.

### 6.3. Demo recording as backup

Record a 60-second screen capture of the Streamlit app: hum recording → transcription → SVG render → arrangement playback. Save as MP4. **This is the unconditional demo-day fallback** in case the live system breaks.

### 6.4. README update

The README should now include:
- Architecture diagram showing 6 stages + optional Stage 7
- License matrix (PESTO CC-BY-NC, MusicGen weights CC-BY-NC, everything else permissive)
- Citation list including YourMT3+ (Chang et al. 2024) and MusicGen (Copet et al. 2023)
- A "Generative AI components" section explicitly listing YourMT3+ and MusicGen and what they do

---

## Updated `requirements.txt` (additions only)

```text
# Stage 2-A: YourMT3+ alternative piano transcriber
# install from: pip install git+https://github.com/mimbres/YourMT3@main
# (check repo at clone time; package may have moved or stabilized)

# Stage 7: MusicGen-Melody arrangement
audiocraft>=1.3.0           # facebookresearch
# audiocraft pulls in encodec automatically

# Confirm transformers can drive the audiocraft text-to-audio pipeline:
transformers>=4.36          # MusicGen support
```

Both fit in the existing Python 3.11 + Torch 2.4+ environment. No CUDA version conflicts on Blackwell with current drivers.

---

## Updated risk table

| Risk | Mitigation |
|---|---|
| YourMT3+ regresses MAESTRO when used as default | Keep ByteDance as default for MAESTRO-style; YourMT3+ engaged only for Romantic-detected pieces via the agent's existing `auto_piano` heuristic. Side-by-side eval before any default flip. |
| MusicGen takes too long for live demo flow | Default the Streamlit tab to musicgen-melody (1.5B, ~5 GB VRAM, ~10 s generation); offer melody-large as a "high quality" toggle for offline use. |
| MusicGen license (CC-BY-NC) blocks commercial reuse | Same constraint as PESTO already in the plan. Document in README. Alternative if it ever matters: MAGNeT (also Meta, also CC-BY-NC, but smaller and faster). |
| Arrangement output sounds bad on a specific user's hum | Provide 6 prompt presets known to work; `temperature=1.0`, `cfg_coef=3.0`, `top_k=250` are good defaults for melody-following. |
| Tuplet pruning regresses snap metric beyond 1pp | Raise `ALLOWED_DENOMS` ceiling to include 24, retest. If still bad, scope the prune to only the rendering path, not the metric path. |
| Voicing hysteresis improves offset20 F1 but regresses no-offset F1 | Decision rule already covers this — only promote if both metrics non-regress. |
| Blackwell drivers incompatible with audiocraft's older torch pin | Force `torch>=2.4` per the requirements; if audiocraft pins older torch, install audiocraft's deps first then override torch. |
| Course presenter on demo day forgets which model is generative | Add a "Pipeline" tab to Streamlit showing model-by-model what's running with a "(generative)" label on YourMT3+ and MusicGen. |

---

## Dependency graph between work items

```
1 (rendering) ───────────────────────────────┐
                                             │
2 (YourMT3+) ────────────┐                   │
                         ├──→ 6.1 re-run gates ──→ 6.2 figures ──→ 6.3 recording ──→ 6.4 README
3 (MusicGen) ────────────┤                   │
                         │                   │
4 (voicing hysteresis) ──┘                   │
                                             │
5 (MedleyDB pseudo-labels) — speculative ────┘
```

Work items 1, 2, 3, 4 are independent and parallelizable. They all feed into 6. Work item 5 is optional and can be skipped without breaking anything.

The only ordering constraint inside the new work: **rendering fixes (item 1) before the demo recording (item 6.3)** — otherwise the recording will show 48-lets on the Bach output.

---

## End of v3.4

Architecture for the existing six stages, validation thresholds for the existing gates, and the agent's already-completed work are all unchanged. v3.4 adds rendering polish, two real generative components, one cleanup experiment for the offset-F1 gap, and a speculative training experiment that can be skipped.

The architecture story is now: *"hybrid pretrained-first pipeline that uses generative seq2seq transcription where it outperforms discriminative methods, classical DP where it doesn't, and generative audio synthesis as an arrangement post-stage"*. That's honest, implementable, and the right shape of project for a Generative AI course.
