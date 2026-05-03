# item-2 — YourMT3+ T5-seq2seq transcription backend (B63)

## Goal
B58 oracle test showed all 18.8pp ASAP upstream loss is in ByteDance.
Per `task_description_v2.md` §Work item 2, integrate YourMT3+ (Chang et al.,
MLSP 2024, [arXiv:2407.04822](https://arxiv.org/abs/2407.04822), Apache-2.0)
as a third option in `humscribe/instrument/`. Default checkpoint: YPTF.MoE+Multi
(noPS) — the same default chosen by the official HF Spaces app.

Pass criteria from the spec:
- MAESTRO F1 ≥ 0.95 (no regression vs ByteDance 0.984) — sanity
- ASAP Bach 5-Fugue mean snap ≥ 0.84 — preserve Bach win
- **ASAP 5-mixed mean snap ≥ 0.74** (vs current 0.590) — main target
- Beethoven snap ≥ 0.92 (vs 0.811, oracle 0.982) — main target
- Schumann snap ≥ 0.93 (vs 0.745, oracle 0.975) — main target
- Liszt snap ≥ 0.20 (vs 0.078) — structural unsalvageable
- **Decision rule**: if Beethoven ≥ 0.85 AND mixed mean ≥ 0.70, promote as
  default for Romantic-detected pieces (auto_piano routing).

## Procedure
1. Cloned the official HF Spaces repo at `huggingface.co/spaces/mimbres/YourMT3`
   to `/workspace/yourmt3_hf`. The MoE-Multi (noPS) checkpoint (542 MB) was
   pulled via `git lfs pull`. Other checkpoints left as LFS pointers to save
   disk.
2. Wrote `humscribe/instrument/yourmt3plus.py` — wraps the Spaces' `model_helper`
   (which itself takes argparse args). Key shims:
   - `_ymt3_cwd()` ctx-manager that sets cwd to the Spaces root and inserts
     `amt/src` + spaces-root on sys.path. The config hardcodes `save_dir='amt/logs'`
     so cwd matters.
   - Use `soundfile` directly to load the audio (the YourMT3+ code uses
     `torchaudio.load(uri=...)` which now requires torchcodec + ffmpeg, not
     installed in this pod).
   - Convert YourMT3+'s internal Note dataclass to our `NoteEvent`; drop
     drum events.
3. `humscribe/config.py:Transcriber` adds `"yourmt3plus"`. `humscribe/pipeline.py`
   updates `auto_piano`: route Romantic-detected pieces (`median_dur > 0.4 s
   AND median_ioi > 0.3 s` from ByteDance output) to YourMT3+ instead of basic_pitch.
4. `scripts/exp_B63_yourmt3_asap.py` runs both backends through the same DP+VT
   pipeline on a 9-piece ASAP set (5 Bach Fugues + 4 Romantic). Beats come
   from the ASAP annotation file (same as `gate_asap_rhythm.py`) so numbers
   compare directly to existing gates.

## Smoke test
BWV 846 single-piece smoke: 17.2 s wall-clock for 735 notes (vs ByteDance 736).
GPU peak ~5 GB at fp16. Apache-2.0 license clean.

## Results

(Filled by `scripts/exp_B63_yourmt3_asap.py` once it finishes — partial table
already showing on Bach Fugues; Romantic pieces in progress.)

Partial results (Bach Fugues, score-beat eval, 5/9 done):
| piece | bd snap | ymt3 snap | Δ |
|---|---|---|---|
| Bach BWV 846 | 0.847 | **0.878** | +3.1pp |
| Bach BWV 848 | 0.851 | **0.927** | +7.6pp |
| Bach BWV 854 | 0.904 | **0.939** | +3.5pp |
| Bach BWV 856 | 0.820 | **0.862** | +4.2pp |
| Bach BWV 857 | 0.873 | **0.885** | +1.2pp |
| 5-Bach mean  | 0.859 | **0.898** | **+3.9pp** |

(Romantic pieces filling in; will populate the full table once B63 finishes.)

## Status
in_progress (waiting for Beethoven/Schumann/Chopin/Liszt)
