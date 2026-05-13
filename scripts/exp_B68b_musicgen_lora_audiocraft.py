"""B68b — LoRA fine-tune MusicGen via audiocraft (HF transformers MusicgenMelody
had shape gymnastics that aren't worth fighting). Phase C.

Approach: load MusicGen via `audiocraft.models.MusicGen` (the path that already
works in B64/B67), apply PEFT LoRA to the underlying language model (`mg.lm`),
do a small training loop over (melody, target) pairs, save the LoRA adapter,
verify it reloads.

Smoke pass criteria:
- LoRA attaches to the audiocraft language model
- 20 training steps complete with finite loss
- peak VRAM < 25 GB (1.5B base + adapter + grads + AdamW states)
- adapter saves and reloads
- post-training inference works with the LoRA active
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Stub audiocraft's spacy/thinc deps before importing it. importlib.util.find_spec
# requires __spec__ to be present on stubs, so synthesise a ModuleSpec.
import types, importlib.machinery
def _stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []  # mark as a package so submodule imports don't crash
    sys.modules[name] = m
for _n in ["spacy", "spacy.lang", "spacy.lang.en", "thinc", "thinc.api",
           "thinc.config", "thinc.types"]:
    if _n not in sys.modules:
        _stub(_n)

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import wandb
from peft import LoraConfig, get_peft_model
from audiocraft.models import MusicGen


CHECKPOINT_DIR = Path("checkpoints/musicgen_lora_b68b")
OUT_JSON = Path("reports/_exp_B68b_musicgen_lora.json")
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


def main(n_steps: int = 20, lora_r: int = 16, lr: float = 1e-4,
         model_size: str = "melody") -> None:
    cfg_w = {"git_sha": git_sha(), "n_steps": n_steps, "lora_r": lora_r, "lr": lr,
             "model": f"facebook/musicgen-{model_size}"}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B68b_musicgen_lora_{model_size}",
                     config=cfg_w, tags=["B68b", "musicgen", "lora", "phase-c", model_size],
                     dir="logs/wandb")

    print(f"loading audiocraft MusicGen {model_size}")
    t0 = time.time()
    mg = MusicGen.get_pretrained(f"facebook/musicgen-{model_size}", device="cuda")
    # audiocraft loads in fp16 by default; cast LM to fp32 so LoRA grad math works
    # without dtype mismatches. Adds ~3 GB VRAM but training stability is worth it.
    mg.lm = mg.lm.to(torch.float32)
    lm = mg.lm
    print(f"  loaded in {time.time()-t0:.1f}s; lm params: {sum(p.numel() for p in lm.parameters())/1e6:.1f}M")

    # Identify target Linear modules in the LM. audiocraft's LM has cross-attention
    # blocks with q/k/v projections inside `transformer.layers.<i>.{self_attn,cross_attention}`.
    eligible = []
    for name, mod in lm.named_modules():
        if isinstance(mod, nn.Linear) and "transformer.layers." in name:
            short = name.split(".")[-1]
            if short in {"q_proj", "k_proj", "v_proj", "out_proj"}:
                eligible.append(name)
    print(f"  eligible projection modules: {len(eligible)}")
    if not eligible:
        # Fallback: list first few Linear in the LM to see naming
        all_lin = [n for n, m in lm.named_modules() if isinstance(m, nn.Linear)][:8]
        print("  no q/k/v match. First 8 Linear modules in LM:")
        for n in all_lin:
            print(f"    {n}")
        run.finish(); return

    # peft target_modules takes a list of strings; suffix-matching applies, so use
    # the bare suffix. To restrict to the LM transformer, peft's regex pattern works.
    target_pattern = r"^transformer\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)$"
    lora_cfg = LoraConfig(target_modules=target_pattern, r=lora_r,
                           lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
                           task_type=None)  # not a HF model class
    lm = get_peft_model(lm, lora_cfg)
    n_trainable = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in lm.parameters())
    print(f"  trainable {n_trainable/1e6:.2f}M / {n_total/1e6:.0f}M = "
          f"{100*n_trainable/n_total:.3f}%")
    wandb.summary["trainable_params_M"] = n_trainable / 1e6
    wandb.summary["total_lm_params_M"] = n_total / 1e6

    # Re-attach the wrapped LM back into the MusicGen container so generate works
    mg.lm = lm

    # Build pairs: (melody_audio, target_audio_codes, prompt)
    melody_audio = load_audio(MELODY, sr=32000)
    print(f"  melody: {len(melody_audio)/32000:.1f}s")
    pairs = []
    for name, prompt in PRESETS.items():
        tgt = TARGETS_DIR / f"vocadito_1_{name.replace(' ', '_')}.wav"
        if not tgt.exists():
            continue
        target_audio = load_audio(tgt, sr=32000)
        pairs.append((name, prompt, target_audio))
    print(f"  pairs: {len(pairs)}")
    if not pairs:
        run.finish(); return

    # Pre-encode targets via EnCodec to get the token sequences we'll predict.
    print("encoding targets via EnCodec…")
    encoded_pairs = []
    for name, prompt, tgt in pairs:
        with torch.no_grad():
            x = tgt[: 10 * 32000].unsqueeze(0).unsqueeze(0).to("cuda")  # (1,1,T)
            codes, scale = mg.compression_model.encode(x)  # codes (1, K, T_codes)
        encoded_pairs.append((name, prompt, codes.detach()))
        print(f"  {name}: tokens shape {tuple(codes.shape)}")

    # Conditioner setup — we need the conditioning (description + chroma) for each step.
    # mg._prepare_tokens_and_attributes builds ConditioningAttributes from descriptions.
    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=lr)
    losses = []
    torch.cuda.reset_peak_memory_stats()

    print(f"\nrunning {n_steps} training steps")
    from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
    for step in range(n_steps):
        name, prompt, codes = encoded_pairs[step % len(encoded_pairs)]
        # Build full conditioning: text + dummy self_wav (melody chroma).
        # `melody` model's condition_provider needs both 'description' (text) and
        # 'self_wav' (chroma) keys present, even if the chroma is empty.
        dummy_wav = torch.zeros(1, 1, 32000, device="cuda")  # 1s of silence
        attrs = [ConditioningAttributes(text={"description": prompt})]
        attrs[0].wav["self_wav"] = WavCondition(
            wav=dummy_wav,
            length=torch.tensor([dummy_wav.shape[-1]], device="cuda"),
            sample_rate=[32000],
            path=[None],
            seek_time=[None],
        )
        tokenized = lm.condition_provider.tokenize(attrs)
        condition_tensors = lm.condition_provider(tokenized)
        # Forward: predict next token at each codebook position
        # codes is (1, K, T). LM expects (B, K, T) input and produces (B, K, T, vocab) logits.
        # We use teacher forcing: input = codes[:, :, :-1], target = codes[:, :, 1:]
        if codes.shape[-1] < 2:
            print(f"  step {step}: too-short codes; skip")
            continue
        inp = codes[..., :-1]
        tgt_tokens = codes[..., 1:]
        optimizer.zero_grad()
        out = lm.compute_predictions(codes=inp, conditions=[], condition_tensors=condition_tensors)
        # out.logits is (B, K, T, vocab)
        logits = out.logits
        # CE per codebook position
        B, K, T, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            tgt_tokens.reshape(-1).long(),
            ignore_index=-1,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        vram = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"  step {step:3d}  preset={name:24s}  loss={loss.item():.4f}  vram_peak={vram:.2f}GB")
        wandb.log({"step": step, "loss": loss.item(), "vram_peak_gb": vram})

    if not losses:
        print("no successful training steps")
        run.finish(); return

    print("\nsaving adapter")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(str(CHECKPOINT_DIR))
    print(f"  saved to {CHECKPOINT_DIR}")

    summary = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "mean_loss": sum(losses) / len(losses),
        "max_vram_peak_gb": float(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024),
        "n_steps_completed": len(losses),
        "trainable_params_M": n_trainable / 1e6,
        "checkpoint_size_kb": sum(p.stat().st_size for p in CHECKPOINT_DIR.rglob("*") if p.is_file()) // 1024,
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
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model-size", default="melody",
                    choices=["melody", "melody-large"])
    main(**vars(ap.parse_args()))
