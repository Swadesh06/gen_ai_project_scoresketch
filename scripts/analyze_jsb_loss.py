"""Quick analysis of the JSB LoRA training loss curve.

Parses logs/exp_C5_jsb_lora.log, splits the per-step losses by prompt
(step % 4), computes per-prompt moving averages, and reports whether
the training has saturated, is still improving, or is OOD/diverging.

Used to decide whether to extend the 1000-step schedule.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np


def main():
    log = Path("logs/exp_C5_jsb_lora.log").read_text()
    # Match "  step  NNN/1000 loss=X.XXXX"
    pat = re.compile(r"step\s+(\d+)/\d+\s+loss=([\d.]+)")
    pairs = [(int(s), float(l)) for s, l in pat.findall(log)]
    pairs = [p for p in pairs if p[0] >= 0]
    if not pairs:
        print("no training-step lines yet"); return
    steps, losses = zip(*pairs)
    steps = np.array(steps); losses = np.array(losses)
    # Group by prompt index (step % 4 because the script rotates 4 prompts)
    by_prompt: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    for s, l in zip(steps, losses):
        by_prompt[s % 4].append(l)
    print(f"N={len(steps)} training-step reports, range steps {steps.min()}..{steps.max()}")
    print(f"\nPer-prompt (step % 4) loss summary:")
    for k in (0, 1, 2, 3):
        vals = by_prompt[k]
        if not vals: continue
        print(f"  prompt {k}: n={len(vals)}  "
              f"min={min(vals):.4f}  median={float(np.median(vals)):.4f}  "
              f"last={vals[-1]:.4f}")

    # Recent trend: last quartile minimum vs first quartile minimum.
    q1 = losses[: len(losses) // 4]
    q4 = losses[3 * len(losses) // 4:]
    print(f"\nOverall trend:")
    print(f"  first quartile: min={q1.min():.4f}  mean={q1.mean():.4f}")
    print(f"  last  quartile: min={q4.min():.4f}  mean={q4.mean():.4f}")
    delta = q4.mean() - q1.mean()
    if delta < -0.1:
        print(f"  -> downward trend ({delta:+.4f}); MORE STEPS may help")
    elif delta > 0.05:
        print(f"  -> upward trend ({delta:+.4f}); likely overfitting or instability")
    else:
        print(f"  -> plateau ({delta:+.4f}); FRESH SCHEDULE or BIGGER r needed")

    # Last 20% steps
    n_recent = len(losses) // 5
    if n_recent >= 4:
        recent = losses[-n_recent:]
        # Did we hit a new min?
        if recent.min() < q1.min():
            print(f"  recent min ({recent.min():.4f}) BELOW first-quartile "
                  f"min ({q1.min():.4f}) — still progressing")


if __name__ == "__main__":
    main()
