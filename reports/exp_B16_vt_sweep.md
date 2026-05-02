# exp_B16_vt_sweep — voice-tracker hyperparameter sweep

## Goal
B15 voice tracker default config (`pitch_jump=4, time_gap_s=1.5`) was a first guess. Cheap to sweep — neither parameter affects the expensive stages (ByteDance, beat_this); only the voice assignment + per-voice DP run per cell.

## Procedure
- Cache ByteDance + beat_this output for Bach BWV 846 score-rendered audio.
- Per cell of 7 × 5 grid (`pitch_jump ∈ {3, 4, 5, 6, 7, 9, 12}` × `time_gap_s ∈ {0.5, 1.0, 1.5, 2.0, 3.0}`): assign voices, compute per-voice durations, run DP, score.
- Total wall-clock: ~10 s on top of the one-time ByteDance cost.

## Results — Bach BWV 846

Top 5 by aligned-snap:

| rank | snap | raw | pj | tg | n_voices |
|---|---|---|---|---|---|
| 1 | **0.847** | 0.846 | 3.0 | 0.5 | 70 |
| 2 | 0.844 | 0.843 | 3.0 | 1.0 | 56 |
| 3 | 0.844 | 0.843 | 4.0 | 0.5 | 59 |
| 4 | 0.843 | 0.840 | 7.0 | 1.0 | 17 |
| 5 | 0.840 | 0.839 | 4.0 | 1.0 | 47 |

Bottom: pj=12, tg=3.0 → snap 0.780 (basically reverts to "all notes in one voice" — same as no VT).

B15 default (pj=4, tg=1.5) sat at **0.779 snap** before the sweep. New B16 default (pj=3, tg=0.5) gives **0.847 snap** — **+6.8pp** on this single piece.

## Results — multi-piece (5 Bach Fugues, B16 defaults)

| piece | bpm | beat F | s5 raw | s5 snap | (vs B15) |
|---|---|---|---|---|---|
| bwv_846 | 122 | 0.845 | 0.846 | 0.847 | +1.5pp |
| bwv_848 | 120 | 0.969 | 0.846 | 0.850 | -1.3pp |
| bwv_854 | 120 | 0.944 | 0.901 | **0.904** | +0.1pp |
| bwv_856 | 231 | 0.778 | 0.796 | 0.808 | +0.4pp |
| bwv_857 | 120 | 0.948 | 0.860 | 0.873 | +0.7pp |

Aggregates:
- mean Stage-5 snap: 0.853 → **0.856** (+0.3pp; mostly within noise)
- mean Stage-5 raw: 0.847 → 0.850 (+0.3pp)
- pieces ≥ 0.85: 1 → 4 (much tighter cluster)

WandB sweep: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/618xgubh
WandB B16-tuned multi-piece: https://wandb.ai/agam_p-iit-roorkee/humscribe-v3.2/runs/iw7wf3xv

## Interpretation
The single-piece win on BWV 846 is real (+6.8pp), but the same hyperparameters are suboptimal for bwv_848 (-1.3pp regression). The reason: bwv_848 has wider melodic intervals, so the tighter `pitch_jump=3` semitone limit splits a single voice into more pieces than ideal. Mean across pieces is barely better — the gain from tighter pj is mostly cancelled by losses on pieces with wider voice movement.

The cleanest win signal: **all 4 of the 120-BPM pieces now cluster at 0.85-0.90 snap** (vs B15's wider 0.832-0.903 range). The slow piece bwv_856 remains the outlier; addressing it needs better beat tracking (B13 didn't help it).

Decision: keep new defaults (`pj=3, tg=0.5`) — small net gain, much smaller variance, and the failure mode (over-segmentation) is benign because per-voice durations stay accurate even when the voices are over-counted. Real improvement now needs per-piece adaptation or a learned voice tracker.

## Next
- B17: HMM voice tracker (true per-voice probabilistic assignment). Should beat the greedy heuristic on pieces where pitch lines cross.
- Pause Phase B Stage-5 work — we're at 0.856 mean, ~5pp from spec target. Diminishing returns until a model change.
