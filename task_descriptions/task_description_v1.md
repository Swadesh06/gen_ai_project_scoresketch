# HumScribe v3.2 — access verification + extra datasets

> **What this is** — Two things you asked for:
> 1. Verification that nothing in the plan is gated, requires institutional email, or has a meaningful access barrier.
> 2. Three small, useful datasets to add: **ASAP** (the right one for rhythm-quantization validation, ironically the dataset ScoreSketch *misused*), **MTG-QBH** (casual a-cappella humming on laptop mics, the closest public match to your demo distribution), and **MIR-1K** (the dataset PESTO trains on — useful as a pitch-tracker sanity check).
>
> **Apply on top of v3 + v3.1.** Architecture, validation thresholds, and roadmap unchanged.

---

## A. Access verification — every asset in the plan

I went through the full v3+v3.1 stack and checked each model and dataset for: HF gating, login wall, license-request email, country/IP restriction, manual-approval queue. Result: **nothing in the primary path is gated.** A few items below have either an automatic-approval click-through or a known download quirk; details are explicit so you know what to expect when you actually run `bootstrap.sh`.

### A.1 Models — access status

| Model | Hosting | Gated? | What you do |
|---|---|---|---|
| **PESTO** | bundled in pip wheel | No | `pip install pesto-pitch` — weights are inside the wheel, no network call at runtime |
| **CREPE / torchcrepe** | auto-download from upstream URL on first call | No | First call to `torchcrepe.predict(..., model='full')` downloads ~88 MB to `~/.cache/torch/hub/`; no auth |
| **ByteDance piano** | Zenodo direct download | No | First `PianoTranscription(...)` call downloads ~330 MB from `https://zenodo.org/record/4034264`; no auth |
| **Basic Pitch** | bundled in pip wheel | No | `pip install basic-pitch` — weights inside the wheel |
| **beat_this** | auto-download from CPJKU lab URL on first call | No | First `File2Beats(checkpoint='final0')` call downloads ~70 MB; no auth |
| **Dynamic HumTrans** | GitHub repo with Git LFS | **Quirk** (not gated, but LFS broken) | Repo is public; weights file may be a 130-byte LFS pointer instead of 150 MB binary. Fix: `git lfs install && git lfs pull` after clone. If LFS still fails (the README admits this), email authors. **This is the only "may need manual contact" model in the plan.** |
| **ROSVOT** | Google Drive link from GitHub README | No (but click-through) | Standard Google Drive shared link — no login needed for the file itself, but verify in the README at clone time |
| **YourMT3+** (Phase-3 stretch) | HuggingFace `mimbres/YourMT3` | No | Public, no gating; ~600 MB |
| **MT3** (alternative) | Magenta GCS bucket | No | Public; T5X format, conversion needed |
| **SwiftF0** (license-clean alternative) | GitHub release | No | Direct binary download |
| **FluidR3_GM SoundFont** | Debian apt repo | No | `sudo apt-get install fluid-soundfont-gm` — done |

**Bottom line for models**: every primary-path model installs without HuggingFace login or any access request. The only friction point in the entire stack is Dynamic HumTrans's LFS situation, which is itself a fallback (not the demo-critical path) — flagged in v3.1 §A.6 already.

### A.2 Datasets — access status

| Dataset | Hosting | Gated / login required? | What you do |
|---|---|---|---|
| **Vocadito** | Zenodo (DOI 10.5281/zenodo.5578807) | No | mirdata loader pulls direct from Zenodo; no auth |
| **MAESTRO v3.0.0** | Magenta GCS bucket | No | mirdata or direct download; no auth |
| **MAPS** | telecom-paristech web form | **Yes — email license request** | Submit form, ~1–3 day turnaround, tied to your email. **Skip this** — MAESTRO covers everything MAPS does and is direct-download |
| **GuitarSet** | Zenodo (DOI 10.5281/zenodo.3371780) | No | mirdata loader; no auth |
| **HumTrans** (literature comparison only) | HuggingFace `dadinghh2/HumTrans` | No (verified — open) | `huggingface-cli download dadinghh2/HumTrans --repo-type dataset` works without login, no Terms-and-Conditions click-through |
| **ASAP (new in v3.2)** | GitHub `CPJKU/asap-dataset` | No | `git clone` direct |
| **MTG-QBH (new in v3.2)** | Zenodo (DOI 10.5281/zenodo.1290712) | No | mirdata loader (`mirdata.initialize("mtg_qbh")`) |
| **MIR-1K (new in v3.2)** | Zenodo (record 3532216) | No | Direct download |
| **URMP** (Phase-3 stretch) | Rochester web form | **Yes — short form, no email approval** | Click through a research-use form; instant download. Skip unless multi-instrument ablation is in scope |

**Bottom line for datasets**: every dataset in the primary plan downloads without manual approval. The only "fill out a form" steps are MAPS (recommend skip, MAESTRO is better anyway) and URMP (Phase-3 stretch only).

### A.3 The HuggingFace authentication situation, in case you bump into it later

Optional setup that costs you nothing on Day 1 and saves a headache if you ever swap in something gated:

```bash
# create free HF account at https://huggingface.co/join
# then on the server:
pip install huggingface_hub
huggingface-cli login
# paste your access token (read-only is fine)
```

This caches your token to `~/.cache/huggingface/token`. Every HF download tool (`datasets`, `huggingface_hub`, `transformers`) will pick it up automatically. If you decide to swap in a gated checkpoint later (some MERT variants are CC-BY-NC and require it), this is one-time setup. **Not required for v3.2's primary path.**

---

## B. Three datasets to add — small, useful, high-ROI

Each is < 200 MB on disk and validates a *specific* component of the pipeline that v3.1's existing four datasets (Vocadito, MAESTRO, GuitarSet, self-collected) don't cover well.

### B.1 ASAP — for rhythm-quantization validation

- **What it is**: 222 MusicXML scores aligned to 1067 piano performances (~92 hours), with **per-beat, per-downbeat, per-time-signature, and per-key-signature annotations** as TSV files, plus note-level alignments from the (n)ASAP TISMIR 2023 paper.
- **Size**: scores + annotations ~50 MB; full dataset including MIDI performances ~1.5 GB. We need only the scores+annotations subset.
- **License**: MIT (per repo).
- **Source**: `https://github.com/CPJKU/asap-dataset` — `git clone` direct.
- **Why it's the right dataset for this purpose**: ScoreSketch trained RhythmT5 on ASAP and that's how it broke (piano IOIs don't match humming IOIs — a tombstone). But ASAP is **exactly the right validation set for a beat-tracking + DP-rhythm-quantization pipeline running on piano audio**, because every performance has hand-checked beat/downbeat/meter annotations against a known score. This is a use-case ASAP was actually designed for, not the misuse that broke ScoreSketch.
- **Concrete use in HumScribe**:
  - **Stage 4 validation** (beat tracking): for any ASAP performance, compare beat_this's predicted beats against ASAP's hand-annotated beats. Compute beat F-measure (mir_eval). This is a much sharper test than the synthetic 60 BPM click track in v3 §7.
  - **Stage 5 validation** (rhythm quantization): given correct beats from ASAP and correct note onsets from a piano transcriber (ByteDance), the Cemgil-Kappen DP should produce note durations that exactly match the MusicXML score's quarterLengths. This is the most direct possible test of whether the DP works — every other input is right, only the quantizer can fail.
  - **Phase 1 sanity gate**: the Bach Inventions in ASAP are short (~2 min), in 4/4, with simple 16th-note rhythms. Pick one, run the full instrument pipeline, compare the produced MusicXML against the ASAP ground-truth MusicXML using a structural diff. **Pass: ≥ 90% of notes match in pitch and quarterLength.**
- **Access in code**:
  ```bash
  cd ~/datasets
  git clone https://github.com/CPJKU/asap-dataset.git asap
  ```
  ```python
  import json
  with open("~/datasets/asap/asap_annotations.json") as f:
      ann = json.load(f)
  # ann["Bach/Fugue/bwv_846/Shi05M.mid"]["performance_beats"] -> [t1, t2, ...]
  ```
- **What's added to validation plan §7 of v3**:

  | Stage | New ASAP test | Pass threshold |
  |---|---|---|
  | 4 | beat_this vs ASAP beats on Bach Fugue BWV 846 | F-measure > 0.90 |
  | 5 | DP quantizer given ASAP beats + ByteDance notes vs ASAP MusicXML quarterLengths | ≥ 90% exact-match |
  | end-to-end (instrument) | full pipeline on ASAP Bach Inventions vs ASAP MusicXML | ≥ 85% pitch + quarterLength match |

### B.2 MTG-QBH — casual a-cappella humming on laptop mics

- **What it is**: 118 sung-melody recordings, 17 subjects, none-to-amateur musicality, recorded on **basic laptop microphones with no post-processing** — explicitly designed to simulate query-by-humming conditions. Total ~50 minutes of audio, average clip ~27 s.
- **Size**: ~150 MB.
- **License**: Free for research use with citation (Salamon et al. 2013); permissive enough for a course project.
- **Source**: Zenodo record `1290712`. Also indexed in mirdata as `mtg_qbh`.
- **Why this is high-ROI**: Vocadito is curated (40 clips, mostly trained singers, careful annotation) and represents an *easier* casual-humming distribution than what your demo will see. MTG-QBH is **closer to the actual deployment chain**: untrained subjects, laptop mics, no editing. If HumScribe clears Vocadito but fails on MTG-QBH, that's a more honest reality check before demo day. The downside: only 118 clips and the original annotations target QBH retrieval, not note transcription — so you can't compute COnP F1 directly without spending some time on note ground-truth.
- **Concrete use in HumScribe**:
  - **Phase 2 qualitative test set**: pick 10 MTG-QBH clips of melodies you know (Twinkle, Frère Jacques, Happy Birthday). Run HumScribe Soft mode. **Visually inspect** the produced SVGs — the rendered melody should be recognizable. This tests "does it work on noisy, casual recordings?" without needing per-clip ground truth.
  - **Optional quantitative**: hand-align ground-truth MIDI for those 10 clips (~10 min/clip in MuseScore as in v3.1 §B.7). Then COnP F1 is computable.
  - **Use as the second hard gate before demo day**: clearing Vocadito + MTG-QBH visual inspection together is a better confidence indicator than Vocadito alone.
- **Access in code**:
  ```python
  import mirdata
  d = mirdata.initialize("mtg_qbh", data_home="~/datasets/mtg_qbh")
  d.download(); d.validate()
  ```

### B.3 MIR-1K — pitch-tracker sanity check on the data PESTO learned from

- **What it is**: 1000 short song clips (4–13 s each), 19 amateur singers, **per-frame pitch contour annotations in semitones** + voiced/unvoiced + lyrics. Total ~133 minutes.
- **Size**: ~1 GB (download only the vocals partition for our use, ~500 MB).
- **License**: Free for research use with citation.
- **Source**: Zenodo `3532216` (`http://mirlab.org/dataset/public/MIR-1K.rar` is the original mirror).
- **Why this matters** (this is subtle but worth doing): PESTO's only released checkpoint is `mir-1k_g7` — trained on MIR-1K. If PESTO's pitch tracking on Vocadito or your own humming looks broken, the question "is PESTO broken or is something upstream broken (resampling, voicing, etc.)?" is hard to answer **unless you have a sanity check on the data PESTO itself trains on**. Running PESTO on MIR-1K and computing Raw Pitch Accuracy (RPA, mir_eval) is that sanity check. Expected: **RPA ≥ 90%** (matches the published PESTO numbers); anything significantly worse indicates a bug in your loading/resampling/voicing pipeline, not in PESTO.
- **Concrete use in HumScribe**:
  - **Stage 2-B.1 unit test**: a sanity check that fires once during Phase-0 setup. Run PESTO on 5 MIR-1K clips with voicing labels, compute RPA. **Pass: RPA > 0.85** (slightly below the published 0.90 to allow for the voicing-window difference).
  - **Diagnostic when humming pipeline misbehaves**: if PESTO's RPA on MIR-1K drops, the bug is in your code, not PESTO. If RPA stays ~0.90 but Vocadito performance is poor, the bug is in the HMM segmenter or downstream — point a debugger there, not at PESTO.
  - **Not needed for demo**, only for debugging.
- **Access in code**:
  ```bash
  cd ~/datasets
  wget -O mir1k.zip "https://zenodo.org/records/3532216/files/MIR-1K.zip?download=1"
  unzip mir1k.zip -d mir1k
  ```
  Pitch annotations are in `mir1k/PitchLabel/{ClipName}.pv` — one MIDI-semitone-or-zero value per 20 ms frame.

### B.4 Updated dataset disk budget

| Dataset | v3.1 budget | v3.2 budget | Required for |
|---|---|---|---|
| Vocadito | 60 MB | 60 MB | Phase 2 quantitative |
| MAESTRO MIDI-only | 80 MB | 80 MB | Reference |
| MAESTRO audio (5 tracks) | 500 MB | 500 MB | Phase 1 quantitative |
| GuitarSet | 3 GB | 3 GB | Phase 1 secondary |
| HumTrans | (skip) | (skip) | Literature only — skip |
| URMP | (skip) | (skip) | Phase 3 stretch — skip unless needed |
| Self-collected | 30 MB | 30 MB | Phase 1+2 demo |
| **ASAP scores+annotations** | — | **50 MB** | **Stage 4+5 validation** |
| **MTG-QBH** | — | **150 MB** | **Phase 2 reality check** |
| **MIR-1K (vocals only)** | — | **500 MB** | **Stage 2-B.1 sanity check** |
| **Total realistic** | ~4 GB | **~4.5 GB** | — |

Still <5 GB. Adding all three of these costs ~700 MB and meaningfully tightens the validation surface.

---

## C. Updates to the bootstrap script

Apply this diff against v3.1 §E `scripts/bootstrap.sh` (just three new mirdata downloads + ASAP clone):

```bash
# 6. download datasets (Vocadito + MAESTRO + ASAP + MTG-QBH + MIR-1K)
echo "=== Downloading evaluation datasets ==="
python - <<'PY'
import mirdata, os, subprocess
home = os.path.expanduser("~/datasets")

# Vocadito — primary humming eval (60 MB)
v = mirdata.initialize("vocadito", data_home=f"{home}/vocadito")
v.download(); v.validate()
print("vocadito ready")

# MAESTRO — piano reference, MIDI-only is fine for now (80 MB)
m = mirdata.initialize("maestro", data_home=f"{home}/maestro", version="3.0.0")
m.download(partial_download=["midi"])
print("maestro MIDI ready")

# MTG-QBH — casual humming reality check (150 MB)
q = mirdata.initialize("mtg_qbh", data_home=f"{home}/mtg_qbh")
q.download(); q.validate()
print("mtg_qbh ready")
PY

# ASAP — rhythm-quantization validation (50 MB scores+annotations only)
if [ ! -d "$HOME/datasets/asap" ]; then
    git clone https://github.com/CPJKU/asap-dataset.git "$HOME/datasets/asap"
fi
echo "asap ready"

# MIR-1K — PESTO sanity-check data (~500 MB)
if [ ! -d "$HOME/datasets/mir1k" ]; then
    cd "$HOME/datasets"
    wget -q --show-progress -O mir1k.zip "https://zenodo.org/records/3532216/files/MIR-1K.zip?download=1"
    unzip -q mir1k.zip -d mir1k && rm mir1k.zip
fi
echo "mir1k ready"
```

---

## D. Three new evaluation scripts (one per dataset)

These slot in next to the three from v3.1 §D.

### D.1 `scripts/eval_asap_rhythm.py`

```python
"""Validate Stage 4 (beat tracking) and Stage 5 (rhythm quantization) on ASAP.
Stage 4 gate: beat F-measure > 0.90 on Bach BWV 846.
Stage 5 gate: ≥ 90% of notes match the score's quarterLength when given correct beats.
"""
import argparse, json
from pathlib import Path
import mir_eval, music21, numpy as np
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.instrument.piano import transcribe_piano

def main(asap_dir, piece_pattern):
    asap = Path(asap_dir).expanduser()
    ann = json.loads((asap / "asap_annotations.json").read_text())
    # find a matching performance
    perf_keys = [k for k in ann if piece_pattern in k]
    if not perf_keys:
        print(f"No piece matching '{piece_pattern}'"); return
    perf_key = perf_keys[0]
    perf = ann[perf_key]
    audio_path = asap / perf_key.replace(".mid", ".wav")  # if audio rendered
    score_xml  = asap / perf_key.rsplit("/", 1)[0] / "xml_score.musicxml"

    # Stage 4: beat tracking vs ASAP beats
    pred_beats, _, _ = track_beats_beat_this(str(audio_path))
    gt_beats = np.array(perf["performance_beats"])
    f_beat = mir_eval.beat.f_measure(gt_beats, pred_beats, f_measure_threshold=0.07)
    print(f"Stage 4 beat F-measure: {f_beat:.3f}  (gate: > 0.90)")

    # Stage 5: DP quantizer given ASAP beats + ByteDance notes
    notes = transcribe_piano(str(audio_path))   # returns NoteEvents
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, gt_beats)
    pred_durations_quarters = (q_off - q_on) / 12.0  # tatums to quarters

    # ground truth quarterLengths from the MusicXML
    score = music21.converter.parse(str(score_xml))
    gt_durations = [n.quarterLength for n in score.flatten().notes]

    n_match = sum(1 for p, g in zip(pred_durations_quarters, gt_durations)
                  if abs(p - g) < 0.05)
    pct = n_match / max(len(gt_durations), 1)
    print(f"Stage 5 quarterLength match: {pct*100:.1f}%  (gate: > 90%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap-dir", default="~/datasets/asap")
    ap.add_argument("--piece-pattern", default="Bach/Fugue/bwv_846")
    main(**vars(ap.parse_args()))
```

### D.2 `scripts/eval_mir1k_pitch_sanity.py`

```python
"""PESTO sanity check on its own training distribution.
Gate: Raw Pitch Accuracy > 0.85 across 5 random MIR-1K clips.
If this fails, the bug is in *our* loading/voicing — not PESTO.
"""
import argparse, random
from pathlib import Path
import mir_eval, numpy as np, soundfile as sf
from humscribe.pitch.pesto_track import track_pitch_pesto

def load_pv(pv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """MIR-1K pitch labels: one MIDI semitone (or 0=unvoiced) per 20 ms."""
    midi = np.array([float(x) for x in pv_path.read_text().split()])
    times = np.arange(len(midi)) * 0.020 + 0.020   # 20 ms hop, 20 ms offset
    return times, midi

def main(mir1k_dir, n_clips):
    root = Path(mir1k_dir).expanduser()
    wavs = list((root / "Wavfile").glob("*.wav"))
    random.seed(0)
    sample = random.sample(wavs, n_clips)
    rpas = []
    for wav in sample:
        audio, sr = sf.read(str(wav))
        if audio.ndim == 2:                     # MIR-1K is stereo: L=accomp, R=vocal
            audio = audio[:, 1]
        gt_t, gt_midi = load_pv(root / "PitchLabel" / wav.with_suffix(".pv").name)
        pred_t, pred_hz, _ = track_pitch_pesto(audio.astype(np.float32), sr)
        pred_cents = 1200 * np.log2(pred_hz / 440.0 + 1e-9) + 6900   # to MIDI*100
        # interpolate to GT timestamps
        pred_at_gt = np.interp(gt_t, pred_t, pred_cents)
        gt_voicing = (gt_midi > 0).astype(bool)
        gt_cents = np.where(gt_voicing, gt_midi * 100, 0.0)
        rpa = mir_eval.melody.raw_pitch_accuracy(
            gt_voicing, gt_cents, gt_voicing, pred_at_gt,
            cent_tolerance=50,
        )
        print(f"{wav.name:30s}  RPA={rpa:.3f}")
        rpas.append(rpa)
    mean = float(np.mean(rpas))
    print(f"\nMean RPA: {mean:.3f}  N={len(rpas)}")
    print(f"GATE: {'PASS' if mean > 0.85 else 'FAIL — fix loading/voicing, not PESTO'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mir1k-dir", default="~/datasets/mir1k")
    ap.add_argument("--n-clips", type=int, default=5)
    main(**vars(ap.parse_args()))
```

### D.3 `scripts/eval_mtg_qbh_visual.py`

```python
"""MTG-QBH qualitative reality check — generate SVGs for 10 well-known clips.
No quantitative gate; you visually inspect the produced scores against the
melodies you recognize. Run before demo day."""
import argparse
from pathlib import Path
import mirdata
from humscribe.pipeline import transcribe
from humscribe.config import PipelineConfig

KNOWN_MELODIES = [   # MTG-QBH track IDs you'll recognize from listening
    # (fill in after running `python -c "import mirdata; ..."` to list track IDs)
    # Pick clips whose target song you know — Twinkle, Yesterday, Frère Jacques, etc.
]

def main(mtg_dir, modes):
    d = mirdata.initialize("mtg_qbh", data_home=mtg_dir)
    tracks = d.load_tracks()
    chosen = KNOWN_MELODIES or list(tracks.keys())[:10]
    for mode in modes.split(","):
        out = Path(f"outputs/mtg_qbh_{mode}")
        out.mkdir(parents=True, exist_ok=True)
        for tid in chosen:
            tr = tracks[tid]
            cfg = PipelineConfig(input_kind="humming", mode=mode)
            r = transcribe(tr.audio_path, cfg)
            (out / f"{tid}.svg").write_text(r.svg)
            print(f"{tid}/{mode}:  notes={r.n_notes}  bpm={r.bpm:.1f}  out={out}/{tid}.svg")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtg-dir", default="~/datasets/mtg_qbh")
    ap.add_argument("--modes", default="soft,medium")
    main(**vars(ap.parse_args()))
```

---

## E. Updated risks (additions only)

| Risk | Mitigation |
|---|---|
| ASAP audio not rendered (some performances are MIDI-only) | The Stage 4+5 validation needs audio; pick ASAP entries that have rendered WAVs in their folder, or render MIDI to WAV with FluidSynth. The Bach Fugue BWV 846 entry has audio available |
| MTG-QBH lacks per-clip note ground truth | Treat as qualitative reality check, not gated quantitative test; if you want numbers, hand-align 5 clips you know in MuseScore (~30 min total) |
| MIR-1K is stereo (L=accomp, R=vocal) and PESTO expects mono | Always extract `audio[:, 1]` (right channel = vocal) before passing to PESTO; the eval script does this |

---

## F. What this means for the v3 day-by-day roadmap

Apply minimally:

- **Day 1 (Phase 0)**: bootstrap downloads ASAP, MTG-QBH, MIR-1K alongside Vocadito + MAESTRO. Adds ~700 MB and ~5 minutes wall time. After bootstrap, run `scripts/eval_mir1k_pitch_sanity.py` as the first end-to-end test that touches a model — if RPA passes, your I/O + PESTO loader is good before any other code is written.
- **Day 3 (Phase 1, end-to-end glue)**: add `scripts/eval_asap_rhythm.py` to the gate before declaring Phase 1 done. The MAESTRO note-F1 > 0.85 gate stays; the ASAP rhythm gate is additive.
- **Day 7 (Phase 2)**: after Vocadito gate, run `scripts/eval_mtg_qbh_visual.py` and visually inspect 10 SVGs. If the recognizable melodies are recognizable, Phase 2 is genuinely done — not just statistically passing on Vocadito.

No day in the roadmap needs to grow; these tests fit inside the existing buffer.

---

## End of v3.2 addendum

Apply on top of v3 + v3.1. Architecture, validation thresholds for the original gates, and roadmap days are unchanged.
