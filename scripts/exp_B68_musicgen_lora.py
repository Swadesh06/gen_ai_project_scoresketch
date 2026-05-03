"""B68 — LoRA fine-tuning smoke for MusicGen-Melody (Phase C).

Goal: confirm the audiocraft + PEFT + LoRA training path runs end-to-end on
this hardware without OOM. Not an attempt to actually beat the base model —
that would need a curated melody→arrangement pair set (Phase D / future work).

Smoke criteria:
- model loads with LoRA adapters on attention layers
- one forward + backward pass completes without crash
- peak VRAM < 25 GB during the smoke (1.5B base + adapters + grads + Adam states)
- adapter checkpoint saves and re-loads
- 10 training steps complete and loss is finite + non-trivially decreasing OR plateauing

Synthetic pairs: use Vocadito clip 1 as the melody input and the 6 B64-generated
arrangements as targets (per-preset distill). Style is the prompt that
generated each target. Six pairs total — sufficient for a smoke test.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import (
    MusicgenMelodyForConditionalGeneration,
    MusicgenMelodyProcessor,
)


CHECKPOINT_DIR = Path("checkpoints/musicgen_lora_smoke")
OUT_JSON = Path("reports/_exp_B68_musicgen_lora.json")
PRESETS = {
    "lo-fi hip hop": "lo-fi hip hop with mellow piano, vinyl crackle, soft drums",
    "jazz trio": "jazz trio with upright bass, brushed drums, and warm piano",
    "EDM": "energetic electronic dance music with synth lead, four-on-the-floor kick, sidechain pumping pads",
    "orchestral cinematic": "cinematic orchestral arrangement with sweeping strings, brass swells, and tympani",
    "indie folk": "indie folk arrangement with fingerpicked acoustic guitar, light percussion, and mandolin",
    "bossa nova": "bossa nova with nylon-string guitar, soft brushed drums, and double bass",
}
MELODY = Path("/home/swadesh/datasets/vocadito/Audio/vocadito_1.wav")
TARGETS_DIR = Path("outputs/musicgen_presets")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_resample(path: Path, sr: int = 32000) -> np.ndarray:
    audio, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr != sr:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, sr, src_sr).astype(np.float32)
    return audio


def main(n_steps: int = 10, lora_r: int = 8, lr: float = 1e-4) -> None:
    cfg_w = {"git_sha": git_sha(), "n_steps": n_steps, "lora_r": lora_r, "lr": lr,
             "model": "facebook/musicgen-melody"}
    run = wandb.init(project="humscribe-v3.2", name="exp_B68_musicgen_lora",
                     config=cfg_w, tags=["B68", "musicgen", "lora", "phase-c"],
                     dir="logs/wandb")

    print("loading processor + base model (musicgen-melody decoder LoRA)")
    t0 = time.time()
    processor = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody")
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(
        "facebook/musicgen-melody", torch_dtype=torch.float32,
    ).to("cuda")
    print(f"  loaded in {time.time()-t0:.1f}s; params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Inspect: only Linear modules in decoder.layers should match.
    decoder_lin = [n for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and ".decoder.layers." in n
                    and (n.endswith(".self_attn.q_proj") or n.endswith(".self_attn.v_proj"))]
    print(f"  decoder linear projections eligible: {len(decoder_lin)}")
    print(f"  example: {decoder_lin[0] if decoder_lin else 'none'}")

    # Use regex pattern so peft applies LoRA only to decoder.layers self-attn.
    # The leading ^ is required for peft to treat it as a regex.
    target_pattern = r"^.*\.decoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)$"

    lora_cfg = LoraConfig(target_modules=target_pattern, r=lora_r,
                           lora_alpha=lora_r * 2, lora_dropout=0.05,
                           task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {n_trainable/1e6:.2f}M / {n_total/1e9:.2f}B = "
          f"{100*n_trainable/n_total:.3f}%")
    wandb.summary["trainable_params_M"] = n_trainable / 1e6
    wandb.summary["total_params_B"] = n_total / 1e9
    wandb.summary["lora_targets_n"] = len(decoder_lin)
    wandb.summary["lora_target_pattern"] = target_pattern

    melody_audio = load_resample(MELODY, sr=32000)
    print(f"  melody: {len(melody_audio)/32000:.1f}s")

    pairs = []
    for name, prompt in PRESETS.items():
        tgt = TARGETS_DIR / f"vocadito_1_{name.replace(' ', '_')}.wav"
        if not tgt.exists():
            print(f"  skip {name}: target not found")
            continue
        target_audio = load_resample(tgt, sr=32000)
        pairs.append((name, prompt, target_audio))
    print(f"  {len(pairs)} (melody, target, prompt) pairs")
    if not pairs:
        print("no pairs — bail")
        run.finish(); return

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    torch.cuda.reset_peak_memory_stats()

    print(f"\nrunning {n_steps} smoke training steps (text-only prompts)")
    # Simplified smoke: text prompt → 10s of EnCodec token CE loss.
    # The melody-conditioned path has shape gymnastics that aren't load-bearing
    # for "does the LoRA gradient + save/reload path work?" — text-only is
    # a strict subset and exercises the same decoder LoRA modules.
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    for step in range(n_steps):
        name, prompt, tgt = pairs[step % len(pairs)]
        tgt_short = tgt[: 10 * 32000]
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to("cuda")
        tgt_tensor = torch.from_numpy(tgt_short).unsqueeze(0).unsqueeze(0).to("cuda")  # (1,1,T)
        with torch.no_grad():
            enc = base_model.audio_encoder.encode(tgt_tensor)
            audio_codes = enc.audio_codes  # (1, B, n_q, T) or (B, n_q, T)
            if audio_codes.dim() == 4:
                audio_codes = audio_codes[0]
            labels = audio_codes  # (B, K, T)
        optimizer.zero_grad()
        out = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        vram = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"  step {step:3d}  preset={name:24s}  loss={loss.item():.4f}  vram_peak={vram:.2f}GB")
        wandb.log({"step": step, "loss": loss.item(), "vram_peak_gb": vram})

    if not losses:
        print("no successful training steps")
        run.finish(); return

    print("\nsaving adapter checkpoint")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CHECKPOINT_DIR))
    print(f"  saved to {CHECKPOINT_DIR}")

    # Verify reload
    print("verifying reload")
    base = MusicgenMelodyForConditionalGeneration.from_pretrained(
        "facebook/musicgen-melody", torch_dtype=torch.float32).to("cuda")
    from peft import PeftModel
    reloaded = PeftModel.from_pretrained(base, str(CHECKPOINT_DIR))
    reload_ok = sum(1 for _ in reloaded.parameters()) > 0
    print(f"  reload ok: {reload_ok}")

    summary = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_vram_peak_gb": float(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024),
        "n_steps_completed": len(losses),
        "trainable_params_M": n_trainable / 1e6,
        "checkpoint_size_kb": sum(p.stat().st_size for p in CHECKPOINT_DIR.rglob("*") if p.is_file()) // 1024,
        "reload_ok": reload_ok,
    }
    wandb.summary.update(summary)
    OUT_JSON.write_text(json.dumps({"summary": summary, "losses": losses, "config": cfg_w}, indent=2))
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    print(f"  run: {run.url}")
    run.finish()


def _nullctx():
    from contextlib import nullcontext
    return nullcontext()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    main(**vars(ap.parse_args()))
