"""C5c: extend C5b r=64 (test loss 0.983) by another 1500 steps.

Loads the C5b PEFT adapter at step_1500, continues training with a fresh
optimizer + cosine schedule over the next 1500 steps. Tests whether
longer training continues to improve the JSB chorale fit.

Saves to checkpoints/musicgen_lora_c5c_jsb/step_<N>/.
"""
from __future__ import annotations
import argparse, json, random, sys, time
import types, importlib.machinery
from pathlib import Path

# audiocraft pulls spacy/thinc — stub them.
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
import torch
import torch.nn.functional as F
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp_C5_jsb_lora import (
    discover_pairs, load_wav, compute_ce_with_mask, rotate_checkpoints,
    SR, PROMPTS,
)


CKPT_DIR = Path("checkpoints/musicgen_lora_c5c_jsb")
SOURCE_ADAPTER = Path("checkpoints/musicgen_lora_c5_jsb/step_1500")
OUT_JSON = Path("reports/_exp_C5c_jsb_lora.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500,
                    help="additional training steps on top of C5b's 1500")
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--model-size", default="melody-large")
    ap.add_argument("--target-dur-s", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.warmup_steps < 0.1 * args.steps
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_ADAPTER.exists():
        raise FileNotFoundError(f"need C5b adapter at {SOURCE_ADAPTER}")

    import wandb
    wandb.init(project="humscribe-v3.2", name="exp_C5c_jsb_lora_r64_extended",
                config=vars(args), tags=["phase-e", "item-5", "genai-musicgen",
                                           "jsb-chorales", "r64", "extended"],
                dir="logs/wandb")

    pairs = discover_pairs(limit=args.limit_pairs,
                            target_dur_s=args.target_dur_s)
    random.shuffle(pairs)
    n_test = max(2, int(args.test_frac * len(pairs)))
    train_pairs = pairs[n_test:]
    test_pairs = pairs[:n_test]
    print(f"pairs: {len(pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")

    from audiocraft.models import MusicGen
    from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition

    print(f"loading MusicGen-{args.model_size}")
    t0 = time.time()
    mg = MusicGen.get_pretrained(f"facebook/musicgen-{args.model_size}",
                                   device="cuda")
    mg.lm = mg.lm.to(torch.float32)
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"({torch.cuda.memory_allocated()/1e9:.2f} GB)")
    # Load C5b adapter (PEFT auto-detects r=64 from saved config).
    mg.lm = PeftModel.from_pretrained(mg.lm, str(SOURCE_ADAPTER), is_trainable=True)
    lm = mg.lm
    n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    print(f"  loaded C5b adapter; trainable {n_train/1e6:.2f}M")
    wandb.summary["trainable_params_M"] = n_train / 1e6
    wandb.summary["source_adapter"] = str(SOURCE_ADAPTER)

    optim = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.steps - args.warmup_steps),
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim, [warmup, cosine], milestones=[args.warmup_steps])

    train_losses = []
    T = int(args.target_dur_s * SR)
    torch.cuda.reset_peak_memory_stats()
    print(f"\nC5c: extending C5b for {args.steps} more steps "
          f"(warmup {args.warmup_steps}, target dur {args.target_dur_s}s)")
    t_start = time.time()

    for step in range(args.steps):
        mel_p, arr_p, _ = train_pairs[step % len(train_pairs)]
        prompt = PROMPTS[step % len(PROMPTS)]
        try:
            mel_a = load_wav(mel_p)
            arr_a = load_wav(arr_p)
        except Exception as e:
            print(f"  step {step}: load failed {e}"); continue
        mel_t = mel_a[:T].unsqueeze(0).unsqueeze(0).to("cuda")
        arr_t = arr_a[:T].unsqueeze(0).unsqueeze(0).to("cuda")
        with torch.no_grad():
            codes, _ = mg.compression_model.encode(arr_t)
        codes = codes.detach()

        attrs = [ConditioningAttributes(text={"description": prompt})]
        attrs[0].wav["self_wav"] = WavCondition(
            wav=mel_t, length=torch.tensor([mel_t.shape[-1]], device="cuda"),
            sample_rate=[SR], path=[None], seek_time=[None])
        tok = lm.condition_provider.tokenize(attrs)
        cond_t = lm.condition_provider(tok)
        out = lm.compute_predictions(codes=codes, conditions=[],
                                       condition_tensors=cond_t)
        pad_mask = torch.ones_like(codes, dtype=torch.bool)
        valid = pad_mask & out.mask
        loss = compute_ce_with_mask(out.logits, codes, valid)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
        optim.step()
        sched.step()
        train_losses.append(float(loss.item()))

        vram = torch.cuda.max_memory_allocated() / 1e9
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}/{args.steps} loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} vram={vram:.2f}GB")
        wandb.log({"step": step, "loss": loss.item(), "vram_peak_gb": vram,
                    "lr": sched.get_last_lr()[0]})

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            ckpt_dir = CKPT_DIR / f"step_{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            lm.save_pretrained(str(ckpt_dir))
            rotate_checkpoints(keep=4)
            print(f"  ckpt: {ckpt_dir}")

    wall_train = time.time() - t_start
    wandb.summary["wall_train_s"] = wall_train

    print(f"\nevaluating on {len(test_pairs)} held-out pairs")
    lm.eval()
    test_losses = []
    with torch.no_grad():
        for mel_p, arr_p, _ in test_pairs:
            try:
                mel_a = load_wav(mel_p); arr_a = load_wav(arr_p)
            except Exception: continue
            mel_t = mel_a[:T].unsqueeze(0).unsqueeze(0).to("cuda")
            arr_t = arr_a[:T].unsqueeze(0).unsqueeze(0).to("cuda")
            codes, _ = mg.compression_model.encode(arr_t)
            attrs = [ConditioningAttributes(text={"description": PROMPTS[0]})]
            attrs[0].wav["self_wav"] = WavCondition(
                wav=mel_t, length=torch.tensor([mel_t.shape[-1]], device="cuda"),
                sample_rate=[SR], path=[None], seek_time=[None])
            tok = lm.condition_provider.tokenize(attrs)
            cond_t = lm.condition_provider(tok)
            out = lm.compute_predictions(codes=codes, conditions=[],
                                           condition_tensors=cond_t)
            pad_mask = torch.ones_like(codes, dtype=torch.bool)
            valid = pad_mask & out.mask
            loss = compute_ce_with_mask(out.logits, codes, valid)
            test_losses.append(float(loss.item()))
    test_mean = float(np.mean(test_losses)) if test_losses else float("nan")
    wandb.summary["test_loss_mean"] = test_mean
    print(f"  test loss mean = {test_mean:.4f} (C5b was 0.983)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "config": vars(args),
        "source_adapter": str(SOURCE_ADAPTER),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "train_loss_min": float(min(train_losses)) if train_losses else None,
        "train_loss_final": float(train_losses[-1]) if train_losses else None,
        "test_loss_mean": test_mean,
        "wall_train_s": wall_train,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
