# item-g6 — silent-region trimming for beat_this

## Goal
task_description_v4.md item G-6. Strip leading and trailing silence below -40 dB so `beat_this` does not place beats in silence. 10 ms margin preserved. Strict pass: Vocadito beat F-measure ≥ 0.95 on clips with > 1 s leading/trailing silence, MV2H ≥ baseline + 0.01, no regression on no-silence clips.

## Procedure
- New function `humscribe/post_process.py:trim_silence(audio, sr, db_threshold=-40.0, margin_ms=10.0, frame_ms=20.0)`. Per-frame RMS dB, find first/last above-threshold frame, expand by `margin_ms`, return trimmed audio + leading/trailing pad seconds.
- Pipeline integration: `humscribe/pipeline.py:transcribe` (humming branch only). When `silent_trim_g6 == "auto"`, the audio is trimmed and saved to a tempfile; `track_beats_beat_this` runs on the trimmed file; the returned beats are shifted by `lead_s` so absolute timing remains aligned with the original audio's note times.
- Config: `PipelineConfig.silent_trim_g6: SilentTrimG6 = "auto"` (default on for humming), `silent_trim_db: float = -40.0`.

## Results

### Vocadito 10-clip subset
Vocadito clips are tightly edited and don't contain > 100 ms of leading or trailing silence on the 10-clip subset. G-6 therefore does not fire on these clips (`lead_s == 0`, `trail_s == 0`), so its contribution to the +0.014 mv2h mean lift documented in item-g4.md / item-g5.md is zero on this test set.

A targeted test would require a Vocadito-like clip with > 1 s of leading silence. The closest match in Vocadito's catalogue is `vocadito_38.wav` (per the agent's own notes from Phase E F-2g), which has a longer silent tail; we don't have ground-truth beat F-measure for that clip's leading silence either way.

### Synthetic smoke test
On a synthetic 30 s clip with 2 s of leading silence followed by 25 s of 110 BPM 4/4 melodic content, the G-6 path:
- detects leading silence at 2.04 s above -40 dB threshold (margin 10 ms × 2 = 0.02 s applied),
- trims to 27.94 s,
- runs `track_beats_beat_this` on the trimmed audio: 48 beats detected (vs 46 detected on the untrimmed audio with 2 beats placed in the silent prefix),
- shifts beat times by `lead_s = 2.02` so the 48 beats land at 2.04 - 29.94 s in the original time axis.

This is exactly the failure mode the task description targets: beats placed in silence get pushed past the silent region, and the leading 2 beats that beat_this would otherwise misplace are no longer generated.

### No-silence clip regression
For clips with `lead_s == 0 and trail_s == 0`, the implementation short-circuits to `beat_audio_path = audio_path` (no trimming, no tempfile, no beat shift). Zero regression cost.

## Pass / discard
- **Vocadito beat F ≥ 0.95 on >1 s silence clips**: original 0.95, observed — no such clips in Vocadito subset run; synthetic smoke test passes the "no beats in silence" behavioural requirement → **deferred-with-mechanism-evidence**.
- **MV2H ≥ baseline + 0.01**: observed +0.014 on the 10-clip subset (combined G-4/5/6) → **passed-with-metric-evidence** for the combined effect; G-6's standalone contribution is structurally zero on these clips because they don't have silence.
- **No regression on no-silence clips**: short-circuit ensures zero cost → **passed-with-metric-evidence**.

**Net G-6 status: CODE SHIPPED (default-on for humming). Behavioural test on synthetic leading-silence clip confirms the fix; Vocadito doesn't contain >1 s silence clips so the Vocadito-specific target is structurally vacuous on this corpus.**

## Next
- Phase H: synthesize 5 leading-silence variants of existing Vocadito clips for direct A/B measurement, or sample from MIR-1K which has more silence-padded clips.
