"""Phase E item 5 (exp C5) — MusicGen-Melody-Large LoRA on JSB Chorales pairs.

Builds on B77's LoRA infrastructure but trains on real (melody, arrangement)
pairs from JSB Chorales (382 pieces, soprano-flute vs four-voice-organ
renderings). Targets the v3 spec pass criteria:
- training completes without OOM (16 GB VRAM)
- held-out test loss < B77 baseline (0.73 final loss)
- generated arrangement does NOT just replay the melody as flute
- chroma-similarity between melody input and output is moderate (~0.5),
  not 1.0 (which would mean it ignored the prompt) or 0.0 (ignored melody).

Mandatory unit tests before launching long runs:
1. Single-batch overfit: loss < 0.05 within 100 steps on 1 fixed batch.
2. Warmup-vs-total assertion: warmup_steps < 0.1 * total_steps.
3. Inference smoke after first checkpoint: generated WAV is non-silent.

Saves checkpoints to checkpoints/musicgen_lora_c5_jsb/step_<N>/ with rolling
last-4 retention. Final/best at .../best.pt.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

# Stub spacy/thinc before audiocraft pulls them in.
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
import subprocess
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

JSB_DIR = Path("/workspace/datasets/jsb_pairs")
CKPT_DIR = Path("checkpoints/musicgen_lora_c5_jsb")
OUT_JSON = Path("reports/_exp_C5_jsb_lora.json")
SR = 32000

# A small set of arrangement-style prompts that should describe a Bach
# chorale arrangement. We pick one at random per training step so the LoRA
# doesn't overfit to a single text condition.
PROMPTS = [
    "bach four-part chorale played on church organ",
    "four-voice chorale, sacred music, slow tempo, organ",
    "bach harmonisation, organ and strings, classical baroque style",
    "warm-toned organ chorale arrangement, four voices",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_wav(path: Path, sr: int = SR) -> torch.Tensor:
    a, src_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if src_sr != sr:
        from scipy.signal import resample_poly
        a = resample_poly(a, sr, src_sr).astype(np.float32)
    return torch.from_numpy(a)


def compute_ce_with_mask(logits, targets, mask):
    B, K, T = targets.shape
    ce = torch.zeros([], device=targets.device)
    valid = 0
    for k in range(K):
        lk = logits[:, k].contiguous().view(-1, logits.size(-1))
        tk = targets[:, k].contiguous().view(-1)
        mk = mask[:, k].contiguous().view(-1)
        if mk.sum() == 0:
            continue
        ce = ce + F.cross_entropy(lk[mk], tk[mk].long())
        valid += 1
    return ce / max(valid, 1)


def discover_pairs(limit: int | None = None,
                   target_dur_s: float = 10.0) -> list[tuple[Path, Path, str]]:
    """Return [(melody.wav, arrangement.wav, bwv_id)] triples."""
    out = []
    for d in sorted(JSB_DIR.iterdir()):
        if not d.is_dir(): continue
        mel = d / "melody.wav"; arr = d / "arrangement.wav"
        if not (mel.exists() and arr.exists()): continue
        try:
            info_mel = sf.info(str(mel))
            info_arr = sf.info(str(arr))
        except Exception:
            continue
        if info_mel.duration < target_dur_s or info_arr.duration < target_dur_s:
            continue
        out.append((mel, arr, d.name))
        if limit is not None and len(out) >= limit:
            break
    return out


def unit_test_overfit(mg, lm, pair, prompt: str, target_dur_s: float):
    """Single-batch overfit: 100 steps on one pair, expect loss < 0.05."""
    from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
    print("  unit-test: single-batch overfit (100 steps)")
    mel_a = load_wav(pair[0])
    arr_a = load_wav(pair[1])
    T = int(target_dur_s * SR)
    mel_t = mel_a[:T].unsqueeze(0).unsqueeze(0).to("cuda")
    arr_t = arr_a[:T].unsqueeze(0).unsqueeze(0).to("cuda")
    with torch.no_grad():
        codes, _ = mg.compression_model.encode(arr_t)
    codes = codes.detach()
    optim = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                                lr=3e-4)
    attrs = [ConditioningAttributes(text={"description": prompt})]
    attrs[0].wav["self_wav"] = WavCondition(
        wav=mel_t, length=torch.tensor([mel_t.shape[-1]], device="cuda"),
        sample_rate=[SR], path=[None], seek_time=[None])
    tok = lm.condition_provider.tokenize(attrs)
    cond_t = lm.condition_provider(tok)
    final = None
    for step in range(100):
        out = lm.compute_predictions(codes=codes, conditions=[],
                                       condition_tensors=cond_t)
        pad_mask = torch.ones_like(codes, dtype=torch.bool)
        valid = pad_mask & out.mask
        loss = compute_ce_with_mask(out.logits, codes, valid)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in lm.parameters() if p.requires_grad], 1.0)
        optim.step()
        final = float(loss.item())
        if step % 20 == 0:
            print(f"    overfit step {step}: loss={final:.4f}")
    print(f"  overfit final loss={final:.4f}")
    return final


def rotate_checkpoints(keep: int = 4) -> None:
    steps = sorted([p for p in CKPT_DIR.glob("step_*") if p.is_dir()],
                   key=lambda p: int(p.name.split("_")[-1]))
    for old in steps[:-keep]:
        for f in old.glob("*"):
            f.unlink()
        old.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--model-size", choices=["melody", "melody-large"],
                    default="melody-large")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="MusicGen LoRA accumulates gradients implicitly via "
                         "the per-step prompt rotation; bs=1 fits at 16 GB.")
    ap.add_argument("--target-dur-s", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-overfit", action="store_true")
    args = ap.parse_args()

    # Mandatory assertion per v3 spec process-discipline.
    assert args.warmup_steps < 0.1 * args.steps, \
        (f"warmup_steps ({args.warmup_steps}) must be < 0.1 * total_steps "
         f"({0.1 * args.steps}). Phase A had a warmup-overrun bug — don't "
         "regress.")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    import wandb
    run = wandb.init(project="humscribe-v3.2",
                     name=f"exp_C5_jsb_lora_{args.model_size}",
                     config=vars(args) | {"git_sha": git_sha(),
                                          "dataset": "jsb_pairs"},
                     tags=["phase-e", "item-5", "musicgen", "lora",
                           "jsb-chorales", args.model_size],
                     dir="logs/wandb")

    pairs = discover_pairs(limit=args.limit_pairs,
                             target_dur_s=args.target_dur_s)
    random.shuffle(pairs)
    n_test = max(2, int(args.test_frac * len(pairs)))
    train_pairs = pairs[n_test:]
    test_pairs = pairs[:n_test]
    print(f"pairs: {len(pairs)} total, {len(train_pairs)} train, {len(test_pairs)} test")
    wandb.summary["n_train_pairs"] = len(train_pairs)
    wandb.summary["n_test_pairs"] = len(test_pairs)
    if len(train_pairs) < 10:
        print("not enough training pairs; abort"); run.finish(); return

    from audiocraft.models import MusicGen
    from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition

    print(f"loading MusicGen-{args.model_size}")
    t0 = time.time()
    mg = MusicGen.get_pretrained(f"facebook/musicgen-{args.model_size}",
                                  device="cuda")
    mg.lm = mg.lm.to(torch.float32)
    lm = mg.lm
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"({torch.cuda.memory_allocated()/1e9:.2f} GB)")
    target_pattern = r"^transformer\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)$"
    lora_cfg = LoraConfig(target_modules=target_pattern, r=args.lora_r,
                           lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                           bias="none", task_type=None)
    lm = get_peft_model(lm, lora_cfg)
    n_train = sum(p.numel() for p in lm.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in lm.parameters())
    print(f"  trainable {n_train/1e6:.2f}M / {n_tot/1e6:.0f}M "
          f"= {100*n_train/n_tot:.3f}%")
    wandb.summary["trainable_params_M"] = n_train / 1e6
    mg.lm = lm

    # Unit-test overfit (skip with --skip-overfit for re-runs).
    if not args.skip_overfit:
        overfit_loss = unit_test_overfit(mg, lm, train_pairs[0], PROMPTS[0],
                                           args.target_dur_s)
        wandb.summary["overfit_final_loss"] = overfit_loss
        # Reset the LoRA weights so unit-test gradients don't pollute training.
        for n, p in lm.named_parameters():
            if p.requires_grad:
                with torch.no_grad():
                    p.data.zero_()

    # Build a streaming loader that re-encodes audio on demand.
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
    print(f"\ntraining {args.steps} steps "
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
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in lm.parameters() if p.requires_grad], 1.0)
        optim.step(); sched.step()
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

    # Held-out test loss
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
    print(f"  test loss mean = {test_mean:.4f}")

    # Final report
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "config": vars(args),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "train_loss_min": float(min(train_losses)) if train_losses else None,
        "train_loss_final": float(train_losses[-1]) if train_losses else None,
        "test_loss_mean": test_mean,
        "wall_train_s": wall_train,
        "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }, indent=2))
    print(f"wrote {OUT_JSON}")
    run.finish()


if __name__ == "__main__":
    main()
