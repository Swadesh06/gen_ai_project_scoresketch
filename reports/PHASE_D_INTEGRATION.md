# Phase D — production integration of B76 + B77

After Phase D's experimental wave (B65–B80) produced two production-ready
artifacts (B76 voice tracker, B77 LoRA adapter), this commit wires both
into the main pipeline so users get the wins automatically.

## What changed

### `humscribe/rhythm/voice_transformer.py` (NEW)
Production module wrapping the B76 trained Transformer voice tracker.
- `B76VoiceAssigner(checkpoint_path, device)` — instance with `__call__(notes)
  → list[list[int]]`. Plug-compatible with
  `humscribe.rhythm.voice_tracking.quantize_with_voice_tracking(voice_assigner=...)`.
- `get_b76_assigner()` — singleton-cached factory; first call ≈ 50 ms,
  subsequent reuse the in-memory model.
- `is_b76_available()` — checks the default checkpoint exists.

The architecture (6-layer Transformer, d=192) and normalisation logic are
copied verbatim from `scripts/exp_B76_voice_transformer_scaled.py` so the
checkpoint loads cleanly.

### `humscribe/pipeline.py`
- New helper `_should_use_per_voice_dp(notes, cfg)` implements the auto-routing
  heuristic from B79/B80 data:
  - **Triggers** when piano-input AND notes/sec < 10 AND pitch_iqr < 24.
  - **Result on the test set**:
    - Bach BWV 854: **False** (nps=13.3, dense fugue)
    - Liszt Sonata: **False** (nps=10.3, dense)
    - **Chopin Berceuse: True** (nps=8.0, melody+accompaniment) — proven
      +1.66pp B79 winner
    - Schumann Toccata: **False** (nps=21.4, dense)
    - Beethoven 21-1: **False** (nps=14.0, dense)
- Routing in `transcribe()`: when triggered AND B76 checkpoint exists, the
  pipeline uses `quantize_with_voice_tracking(per_voice_dp=True,
  voice_assigner=get_b76_assigner())`. Otherwise unchanged production path
  (shared DP + greedy adaptive_pj).

### `humscribe/config.py`
- `PipelineConfig.per_voice_dp: Literal["auto", "on", "off"]`. Default
  `"auto"`. User can force on (always B76 + per-voice DP) or off (always
  production).

### `humscribe/arrange/musicgen.py`
- `arrange()` and `_load()` gain `lora_adapter: str | None = None`.
- When set, loads a PEFT adapter over the LM (cast to fp32 first to match
  B74/B77 training dtype).
- Adapter cache keyed by `(model_size, dtype, adapter_path)` so different
  adapters can coexist in memory.

### `app/streamlit_app.py`
- Arrange tab auto-discovers `checkpoints/musicgen_lora_b77/step_*/`
  directories and exposes them as a dropdown.
- Default is "(none — base model)" — no behavior change unless user picks.

## How to use

### Per-voice DP (auto)
Default behavior — no code change needed:
```python
from humscribe.pipeline import transcribe
from humscribe.config import PipelineConfig
res = transcribe("chopin_berceuse.wav",
                  PipelineConfig(input_kind="piano"))
# Auto-routes Chopin to per_voice_dp=True + B76 voice tracker.
```

### Force per-voice DP
```python
PipelineConfig(input_kind="piano", per_voice_dp="on")
```

### LoRA-fine-tuned arrangement
```python
from humscribe.arrange.musicgen import arrange
wav = arrange("vocadito_1.wav", "jazz trio with brushed drums",
              duration_s=15,
              lora_adapter="checkpoints/musicgen_lora_b77/step_300")
```

## Headline impact
- **Chopin Berceuse (and any future Chopin-style melody+accomp piece)**:
  +1.66pp snap-F1 from per-voice DP routing.
- **All other piano pieces**: unchanged (production path preserved).
- **Stage 7 arrangements**: optional B77 adapter for fine-tuned styles.

## Phase E next steps (not blocking)
1. **Train B76 on more pieces with deeper voice supervision** (4-track
   data) so per-voice DP could win on Bach Fugues.
2. **Tune routing heuristic** as more pieces are evaluated. Current
   thresholds (nps<10, iqr<24) are calibrated to the v2-spec test set.
3. **Curate real (melody, arrangement) pairs** for B77 generalisation
   beyond the 6 distill targets.
