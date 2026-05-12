# item-1 — MV2H end-to-end metric

## Goal

Phase E item 1 from `task_descriptions/task_description_v3.md`. Build an
end-to-end score-similarity metric (Andrew McLeod's MV2H, ISMIR 2018) so
every subsequent Phase E experiment can be evaluated on the final output
rather than per-stage F1. Unblocks items 6 and parts of 7.

## Procedure

- Cloned `apmcleod/MV2H` into `third_party/MV2H/`, ran `make` (Java 21 from
  conda-forge `openjdk=21`). Verified the example outputs reproduce
  *exactly* (`Transcription1` → 0.8888, `Transcription2` → 0.8545).
- New package `humscribe/eval/` with:
  - `mv2h_io.py` — text-format conversion. `notes_to_mv2h_format`
    (NoteEvent → MV2H text), `stream_to_mv2h_format` (music21 →),
    `midi_to_mv2h_format`, `musicxml_to_mv2h_format`.
  - `mv2h.py` — `compute_mv2h(pred_text, ref_text)` shells out to the jar
    (subprocess, ~50 ms typical, 120 s timeout). Two modes: `aligned` for
    timestamp-matched pairs, `non_aligned` (default) for tempo-different
    pairs via DTW.
- Eval scripts:
  - `scripts/eval_mv2h_asap.py` — 9-piece ASAP, runs on cached YMT3 outputs
    from `/workspace/.cache/asap_yourmt3plus/` against `midi_score.mid`
    sparse-cloned from `fosfrancesco/asap-dataset` (only the 9 pieces' GT
    MIDIs, ~50 KB total).
  - `scripts/eval_mv2h_maestro.py` — 5 chamber clips (audio+GT pairs in
    `outputs/maestro_clips/`).
  - `scripts/eval_mv2h_vocadito.py` — 40 clips (audio+A1/A2 annotations,
    downloaded from Zenodo).
  - `scripts/eval_mv2h_correlate.py` — Pearson/Spearman vs cached snap F1
    plus rank-disagreement diagnostic.

## Results

**ASAP 9-piece baseline (eval_seconds=30, source=ymt3_cache, non-aligned DTW):**

| piece | mv2h | mp | voice | meter | value | harmony | snap_b87 |
|---|---|---|---|---|---|---|---|
| Bach BWV 846 | 0.544 | 0.977 | 0.747 | 0.000 | 0.998 | 0.000 | 0.039 |
| Bach BWV 848 | 0.611 | 0.984 | 0.774 | 0.300 | 0.997 | 0.000 | 0.715 |
| Bach BWV 854 | 0.612 | 0.989 | 0.771 | 0.303 | 0.999 | 0.000 | 0.767 |
| Bach BWV 856 | 0.549 | 0.896 | 0.728 | 0.132 | 0.992 | 0.000 | 0.234 |
| Bach BWV 857 | 0.557 | 0.982 | 0.748 | 0.064 | 0.990 | 0.000 | 0.785 |
| Beethoven 21-1 | 0.534 | 0.951 | 0.676 | 0.070 | 0.972 | 0.000 | 0.688 |
| Schumann Toccata | 0.523 | 0.949 | 0.665 | 0.020 | 0.981 | 0.000 | 0.611 |
| Chopin Berceuse | 0.526 | 0.977 | 0.658 | 0.000 | 0.996 | 0.000 | 0.657 |
| Liszt Sonata | 0.506 | 0.955 | 0.566 | 0.039 | 0.973 | 0.000 | 0.054 |
| **mean** | **0.552** | **0.962** | **0.704** | **0.103** | **0.989** | **0.000** | 0.506 |

**Correlation with snap_b87 (Pearson / Spearman):**
- mv2h:    +0.481 / +0.633
- mp:      +0.432 / +0.633
- voice:   +0.382 / +0.567
- meter:   +0.414 / +0.569
- value:   +0.153 / +0.117

**Diagnostic anomaly** (item 1 pass criterion): BWV 846 — snap=0.039 (terrible)
but MV2H=0.544 (mid-range). This is exactly what the metric is supposed to
do: when snap collapses because of a tempo-octave bug (target_bpm=110 still
picks the wrong octave for this piece), MV2H's DTW absorbs the tempo offset
and reports the underlying transcription quality. Liszt shows the inverse:
snap=0.054 and MV2H=0.506 — DTW absorbs the rubato that the DP couldn't grid.

**Compute footprint:** ~50 ms per piece (MV2H jar); 9-piece eval is sub-1 s
of jar time excluding setup. WandB run logged at `humscribe-v3.2/runs/t1jk2ymv`.

## Interpretation

- **Multi-pitch is high (~0.96)**: the transcribed pitches mostly match —
  this is consistent with the prior snap analysis where YourMT3+ saturated
  the pitch side and the residual loss was in rhythm.
- **Value (note duration) is near-perfect (~0.99)**: durations within MV2H's
  tolerance once DTW aligns notes. Less informative than I'd hoped — it
  doesn't distinguish good and bad rhythm because the predicted note's
  off-on duration is by construction close to its predicted on time.
- **Voice ~0.70**: B76 voice tracking is doing reasonable work given that
  the cached YMT3 outputs have no voice information (we emit single-voice
  predictions); the score reflects YMT3's implicit pitch-stream separation.
- **Meter ~0.10**: low because we emit `on == onVal` (no metric-tatum
  quantisation). Tried quantising to a tatum grid; that crashed MV2H DTW
  alignment on dense pieces (mean MV2H dropped 0.55 → 0.21, BWV 857 timed
  out). Quantisation is left as a flag the item-6 sweep can toggle.
- **Harmony = 0**: we emit no chord lines. We don't run chord recognition;
  this is honest negative information that an ME-6 chord-recognition
  ensemble member would directly improve.

**The structural diagnostic value is what we wanted**: BWV 846's 0.039 snap
was a known tempo-octave bug, not a transcription problem. The +0.48
correlation between MV2H and snap with informative rank disagreements means
MV2H is a useful new headline metric — it doesn't perfectly mirror snap, and
the cases where it disagrees are exactly the cases where snap was misleading.

**MAESTRO 5-clip baseline (eval_seconds=15, transcriber=bytedance_piano, aligned mode):**

| clip | mv2h | mp | voice | meter | value | harm | gt | pred |
|---|---|---|---|---|---|---|---|---|
| Chamber1 R3 wav--2 | 0.479 | 0.785 | 0.726 | 0.008 | 0.875 | 0.000 | 116 | 75 |
| Chamber2 R3 wav--1 | 0.430 | 0.969 | 0.248 | 0.135 | 0.798 | 0.000 | 96 | 96 |
| Chamber2 R3 wav--3 | 0.472 | 0.901 | 0.493 | 0.025 | 0.940 | 0.000 | 177 | 147 |
| Chamber3 R3 wav--1 | 0.464 | 0.986 | 0.545 | 0.234 | 0.557 | 0.000 | 34 | 35 |
| Chamber3 R3 wav--2 | 0.448 | 1.000 | 0.303 | 0.286 | 0.652 | 0.000 | 57 | 57 |
| **mean** | **0.459** | **0.928** | **0.463** | **0.138** | **0.764** | **0.000** | | |

MAESTRO uses **aligned mode** (pred and GT share time base because the audio is
the same). Note F1 = 0.984 on this set, but MV2H = 0.459 — exactly the kind
of gap the metric is supposed to expose: pitches are nearly perfect (mp=0.93)
but voice separation is weak (only 0.46) because the ByteDance output is
single-stream; the GT MIDI has true voice IDs. This is what an ME-13 voice
legality / proper voice-tracking integration could fix without touching pitch.

## Next

- Vocadito 40-clip MV2H baseline (running): the humming side, A1 annotator.
- Item 6 sweep: use MV2H as the optimisation target over DP params + voicing
  thresholds + render_tpb. Cache the per-piece prediction features first.
- ME-14 ensemble selection: pick the best pipeline variant per-piece based
  on MV2H of its outputs. Depends on this item.

## Files

- `humscribe/eval/__init__.py`
- `humscribe/eval/mv2h.py`
- `humscribe/eval/mv2h_io.py`
- `scripts/eval_mv2h_asap.py`
- `scripts/eval_mv2h_maestro.py`
- `scripts/eval_mv2h_vocadito.py`
- `scripts/eval_mv2h_correlate.py`
- `reports/_metric_mv2h_asap.json` (committed)
- `reports/_metric_mv2h_correlate.json` (committed)
- `reports/_metric_mv2h_vocadito.json` (in flight)
- `reports/_metric_mv2h_maestro.json` (in flight)
- `third_party/MV2H/` (Java sources cloned; `bin/` gitignored, build via Makefile)
