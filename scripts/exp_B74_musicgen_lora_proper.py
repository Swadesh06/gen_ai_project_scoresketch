"""B74 — MusicGen LoRA fine-tune with PROPER delay-pattern mask (Phase D).

B68b validated the LoRA path but loss was NaN because the EnCodec target
tokens include delay-pattern positions that should be ignored. The proper
recipe (per `audiocraft/solvers/musicgen.py:_compute_cross_entropy`):

```
model_output = lm.compute_predictions(audio_tokens, [], condition_tensors)
mask = padding_mask & model_output.mask    # delay-pattern aware
ce, ce_per_codebook = compute_cross_entropy(model_output.logits, audio_tokens, mask)
```

Where the cross entropy is averaged over K codebooks of CE on logits[mask]
vs targets[mask].

This script does a real (multi-hundred-step) LoRA fine-tune on a synthetic
distill task: melody = Vocadito clip 1, target = the 6 B64 / B71 generated
arrangements. The "distill from base" should converge to the base model's
own behavior — useful sanity for the training path.

Settings:
- 200 training steps
- LoRA r=16, attached to decoder self_attn + cross_attention q/k/v/out
- AdamW lr=1e-4, cosine schedule
- Batch size = 1 clip pair per step (sequential)
- fp32 LM (~5.5 GB) + adapters + grads + AdamW = ~15 GB peak
- Save adapter every 50 steps, plus final
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import types, importlib.machinery
def _stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    sys.modules[name] = m
for _n in ["spacy", "spacy.lang", "spacy.lang.en", "thinc", "thinc.api",
           "thinc.config", "thinc.types"]:
    if _n not in sys.modules:
        _stub(_n)

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from peft import LoraConfig, get_peft_model
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition

CHECKPOINT_DIR = Path("checkpoints/musicgen_lora_b74")
OUT_JSON = Path("reports/_exp_B74_musicgen_lora_proper.json")
PRESETS = {
    "lo-fi hip hop": "lo-fi hip hop with mellow piano, vinyl crackle, soft drums",
    "jazz trio": "jazz trio with upright bass, brushed drums, and warm piano",
    "EDM": "energetic electronic dance music with synth lead, four-on-the-floor kick",
    "orchestral cinematic": "cinematic orchestral arrangement with sweeping strings",
    "indie folk": "indie folk arrangement with fingerpicked acoustic guitar",
    "bossa nova": "bossa nova with nylon-string guitar, soft brushed drums",
}
MELODY = Path("/home/swadesh/datasets/vocadito/Audio/vocadito_1.wav")
TARGETS_DIR = Path("outputs/musicgen_presets")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_audio(path: Path, sr: int = 32000) -> torch.Tensor:
    audio, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr != sr:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, sr, src_sr).astype(np.float32)
    return torch.from_numpy(audio)


def compute_ce_with_mask(logits, targets, mask):
    """Per-codebook CE, averaged. Same recipe as audiocraft's solver."""
    B, K, T = targets.shape
    ce = torch.zeros([], device=targets.device)
    valid_pos = 0
    for k in range(K):
        logits_k = logits[:, k].contiguous().view(-1, logits.size(-1))
        targets_k = targets[:, k].contiguous().view(-1)
        mask_k = mask[:, k].contiguous().view(-1)
        if mask_k.sum() == 0:
            continue
        ce_targets = targets_k[mask_k].long()
        ce_logits = logits_k[mask_k]
        ce_k = F.cross_entropy(ce_logits, ce_targets)
        ce = ce + ce_k
        valid_pos += 1
    return ce / max(valid_pos, 1)


def main(n_steps: int = 200, lora_r: int = 16, lr: float = 1e-4,
         model_size: str = "melody", save_every: int = 50,
         eval_every: int = 50, target_dur_s: float = 10.0) -> None:
    cfg_w = {"git_sha": git_sha(), "n_steps": n_steps, "lora_r": lora_r, "lr": lr,
             "model": f"facebook/musicgen-{model_size}",
             "save_every": save_every, "target_dur_s": target_dur_s}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B74_musicgen_lora_proper_{model_size}",
                     config=cfg_w, tags=["B74", "musicgen", "lora", "phase-d", model_size],
                     dir="logs/wandb")

    print(f"loading audiocraft MusicGen {model_size}")
    t0 = time.time()
    mg = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}", device="cuda")
    mg.lm = mg.lm.to(torch.float32)
    lm = mg.lm
    print(f"  loaded in {time.time()-t0:.1f}s; lm params: {sum(p.numel() for p in lm.parameters())/1e6:.1f}M")

    target_pattern = r"^transformer\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)$"
    lora_cfg = LoraConfig(target_modules=target_pattern, r=lora_r,
                           lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
                           task_type=None)
    lm = get_peft_model(lm, lora_cfg)
    n_trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in lm.parameters())
    print(f"  trainable {n_trainable/1e6:.2f}M / {n_total/1e6:.0f}M = "
          f"{100*n_trainable/n_total:.3f}%")
    wandb.summary["trainable_params_M"] = n_trainable / 1e6
    mg.lm = lm

    melody_audio = load_audio(MELODY, sr=32000)
    print(f"  melody: {len(melody_audio)/32000:.1f}s")

    pairs = []
    for name, prompt in PRESETS.items():
        tgt = TARGETS_DIR / f"vocadito_1_{name.replace(' ', '_')}.wav"
        if not tgt.exists():
            continue
        target_audio = load_audio(tgt, sr=32000)
        pairs.append((name, prompt, target_audio))
    print(f"  {len(pairs)} pairs")
    if not pairs:
        run.finish(); return

    print("encoding targets via EnCodec…")
    encoded_pairs = []
    target_T = int(target_dur_s * 32000)
    for name, prompt, tgt in pairs:
        with torch.no_grad():
            x = tgt[:target_T].unsqueeze(0).unsqueeze(0).to("cuda")
            codes, _ = mg.compression_model.encode(x)
        encoded_pairs.append((name, prompt, codes.detach()))

    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                                    lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    losses = []
    torch.cuda.reset_peak_memory_stats()

    print(f"\nrunning {n_steps} training steps")
    dummy_wav = torch.zeros(1, 1, 32000, device="cuda")
    for step in range(n_steps):
        name, prompt, codes = encoded_pairs[step % len(encoded_pairs)]
        attrs = [ConditioningAttributes(text={"description": prompt})]
        attrs[0].wav["self_wav"] = WavCondition(
            wav=dummy_wav,
            length=torch.tensor([dummy_wav.shape[-1]], device="cuda"),
            sample_rate=[32000], path=[None], seek_time=[None],
        )
        tokenized = lm.condition_provider.tokenize(attrs)
        condition_tensors = lm.condition_provider(tokenized)
        # Forward via compute_predictions — returns logits + delay-pattern mask
        out = lm.compute_predictions(codes=codes, conditions=[],
                                       condition_tensors=condition_tensors)
        # padding mask (all valid for fixed-length) AND model's delay-pattern mask
        padding_mask = torch.ones_like(codes, dtype=torch.bool)
        valid_mask = padding_mask & out.mask
        loss = compute_ce_with_mask(out.logits, codes, valid_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        sched.step()
        losses.append(float(loss.item()))
        vram = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}/{n_steps}  preset={name:24s}  loss={loss.item():.4f}  vram={vram:.2f}GB  lr={sched.get_last_lr()[0]:.2e}")
        wandb.log({"step": step, "loss": loss.item(), "vram_peak_gb": vram,
                    "lr": sched.get_last_lr()[0]})

        if (step + 1) % save_every == 0 or step == n_steps - 1:
            ckpt_dir = CHECKPOINT_DIR / f"step_{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            lm.save_pretrained(str(ckpt_dir))
            print(f"  ckpt saved: {ckpt_dir}")

    if not losses:
        print("no successful training steps")
        run.finish(); return

    summary = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "mean_loss": sum(losses) / len(losses),
        "max_vram_peak_gb": float(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024),
        "n_steps_completed": len(losses),
        "trainable_params_M": n_trainable / 1e6,
        "loss_decay_pct": ((losses[0] - losses[-1]) / max(losses[0], 1e-6)) * 100.0,
    }
    wandb.summary.update(summary)
    OUT_JSON.write_text(json.dumps({"summary": summary, "losses": losses, "config": cfg_w}, indent=2))
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model-size", default="melody",
                    choices=["melody", "melody-large"])
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--target-dur-s", type=float, default=10.0)
    main(**vars(ap.parse_args()))
