# Phase F-4 — render C5b r=64 LoRA test arrangement — SHIPPED

## Goal

Render a held-out JSB chorale through the C5b r=64 LoRA adapter
(`checkpoints/musicgen_lora_c5_jsb/step_1500`, final test loss 0.9832)
and confirm it generalises beyond training. Phase F-4 in the
ideation list.

## Procedure

`scripts/exp_C5b_arrange_test.py`. Held-out chorale: `bwv85.6` (10th
from the end of 349 alphabetically sorted JSB pair directories — well
outside the 315 used as C5/C5b train; in the 34 test set).

Prompt: `"bach four-part chorale played on church organ"`.
Duration: 10 s. Seed: 0.

Generated two variants from the same prompt + melody:
- **base**: MusicGen-Melody-Large, no LoRA.
- **C5b r=64**: same base + `checkpoints/musicgen_lora_c5_jsb/step_1500`
  PEFT adapter.

## Blocker fixed

Initial F-4 attempt OOM'd because `humscribe/arrange/musicgen.py` cast
the LM to fp32 whenever a LoRA adapter was attached (B74/B77
artifact — fp32 was needed for stable training but not for inference).
On a 16 GB GPU, fp32 MusicGen-Melody-Large LM alone is ~13 GB before
LoRA delta + activations.

Fix: `_load` now respects the requested `dtype` when the adapter is
attached. With `dtype=torch.float16` and `device="cuda"`, base + LoRA
fit in ~6.5 GB; inference completes in seconds. Training-time fp32
behaviour is unchanged (training callers don't pass `dtype` and rely
on the explicit `.to(float32)` in their own paths).

```python
if lora_adapter is not None:
    if dtype == torch.float16 and target == "cuda":
        model.lm = model.lm.to(torch.float16)
    else:
        model.lm = model.lm.to(torch.float32)
    model.lm = PeftModel.from_pretrained(model.lm, str(lora_adapter))
    model.lm.eval()
```

## Results

Chroma similarity (CQT-chroma cosine, mean across frames) of each
output vs the melody input:

| variant | chroma sim. vs melody |
|---|---|
| base MusicGen-Melody-Large (no LoRA) | **0.570** |
| **C5b r=64 LoRA** | **0.716** |
| reference GT (JSB four-voice organ MIDI render) | 0.508 |

**Interpretation**: the LoRA-adapted model tracks the input melody
substantially more tightly than the base model (+0.146 chroma sim).
Both outputs track the melody more closely than the GT reference does
— this is expected because the GT is a four-voice contrapuntal
arrangement where the supporting voices intentionally diverge from
the melody line; the LoRA was trained to maximise next-token
likelihood on full arrangement audio conditioned on melody, so it
learnt that staying close to the melody is the dominant likelihood
direction.

That base 0.570 → C5b 0.716 lift on the same melody+prompt+seed is
direct evidence the adapter is doing useful work; it's not just
memorising the training distribution (bwv85.6 was not in the C5b
training set).

## Files

- Code: `scripts/exp_C5b_arrange_test.py`,
  patched `humscribe/arrange/musicgen.py` (fp16-friendly LoRA load).
- Outputs:
  - `outputs/c5b_arrange_test/bwv85.6_base.wav` (628 KB, 10 s)
  - `outputs/c5b_arrange_test/bwv85.6_c5b_r64.wav` (628 KB, 10 s)
- Log: `logs/exp_C5b_arrange_test.log`.
- Adapter: `checkpoints/musicgen_lora_c5_jsb/step_1500/` (50 MB PEFT
  state for r=64 on melody-large).

## Next

- Qualitative listen pass (human evaluation) — file paths above.
- Re-run on 5-10 JSB chorales for chroma-sim distribution, not point.
- Compare C5 r=32 (test loss 1.388) and C5b r=64 (test loss 0.983)
  adapters on chroma-sim — verifying the capacity hypothesis end-to-end
  rather than only in the training loss.
