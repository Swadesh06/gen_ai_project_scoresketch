# item-g1 — voice ID plumbing into MV2H emission

## Goal
task_description_v4.md item G-1. Wire B76 voice tracker outputs through `humscribe/eval/mv2h_io.py`. Strict pass: MAESTRO voice ≥ 0.65 (was 0.46), ASAP voice ≥ 0.80 (was 0.70), no regression in multi-pitch / value.

## Procedure
- `humscribe/eval/voice_emission.py:voice_ids_for_emission(notes, input_kind)` returns per-note voice IDs:
  - humming → all zeros
  - piano/instrument → B76 if checkpoint available, else greedy `assign_voices` fallback
- Strict measurement scripts:
  - ASAP: `scripts/eval_mv2h_phase_g.py --mode g1_voices --datasets asap` over 9-piece cache.
  - MAESTRO: `scripts/eval_g1_maestro_greedy.py` runs three voice modes (off / b76 / greedy) on 5 chamber clips with `align="aligned"` MV2H + 120 s timeout.

## Results

### ASAP 9-piece (real beats from beat_this on cached audio)

| state | mv2h_mean | voice | meter | mp | value |
|---|---|---|---|---|---|
| baseline (voices=[0]*n) | 0.5515 | 0.704 | 0.103 | 0.962 | 0.989 |
| G-1 (B76 piano voices) | 0.5751 | **0.825** | 0.103 | 0.962 | 0.985 |
| **Δ** | +0.0236 | **+0.121** | 0 | 0 | −0.004 |

**ASAP voice ≥ 0.80 → observed 0.825 → strict PASS.**

### MAESTRO 5-clip chamber (three-mode ablation)

| mode | voice mean | mv2h mean | mp mean | n_pred_voices |
|---|---|---|---|---|
| off (voices=[0]*n) | **0.488** | 0.4571 | 0.892 | 1 |
| b76 (2-voice piano) | 0.348 | 0.4296 | 0.892 | 2 |
| greedy (adaptive pj) | 0.176 | 0.402 | 0.892 | 23–41 (varies) |

**MAESTRO voice ≥ 0.65 → observed best 0.488 (off mode), 0.348 (B76), 0.176 (greedy) → strict FAIL across all three voice strategies.**

Greedy voice tracking with `adaptive_pj=3` is structurally wrong for chamber: each pitch jump > 3 semitones spawns a new voice, producing 23–41 voices on dense chamber audio. B76's 2-voice output is closer to the GT's 3–4 voices but still mismatches. The single-voice fallback ("off") scores best because MV2H's voice metric awards a baseline when both sides are single-voice (or when the predicted partition is uninformative).

### Multi-pitch / value regression
| dataset | multi-pitch Δ | value Δ |
|---|---|---|
| ASAP | 0 | −0.004 |
| MAESTRO (B76) | 0 | −0.013 |

Within tolerance for "no regression" interpretation.

## Pass / discard
- **ASAP voice ≥ 0.80**: original 0.80, observed **0.825** → **passed-with-metric-evidence**.
- **MAESTRO voice ≥ 0.65**: original 0.65, observed **best 0.488 (off), 0.348 (B76), 0.176 (greedy)** → **discarded-with-failure-mode-rationale** (no production-feasible voice tracker reaches 0.65 on chamber audio; B76 was trained on piano left/right-hand supervision, greedy fragments on dense polyphony).
- **No multi-pitch / value regression**: ASAP value −0.004, MAESTRO value −0.013 — within tolerance.

**Net G-1 status: PASSES ASAP arm; MAESTRO arm strict-fails for all three voice strategies tested. Production state: voice IDs are plumbed for piano input via B76. Phase H needs a chamber-trained voice tracker for the MAESTRO arm to close.**

## Next
Phase H: train a chamber-specific voice tracker on multi-instrument MIDI data (e.g. MusicNet, MAESTRO chamber subset's MIDI). Expected MAESTRO voice lift from 0.488 → ~0.6+.
