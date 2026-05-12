"""C5b r=64 chroma similarity distribution on 5 held-out chorales.

Distribution check — verify the +0.146 lift on bwv85.6 isn't a one-off.
(C5 r=32 weights were overwritten in-place by C5b training; only C5b
remains, so we report base vs C5b only.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.arrange.musicgen import arrange_to_file


JSB_PAIRS = Path("/workspace/datasets/jsb_pairs")
OUT_DIR = Path("outputs/c5_vs_c5b_multi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# C5 r=32 final adapter (test loss 1.388)
ADAPTER_C5 = "checkpoints/musicgen_lora_c5_jsb_b/step_900"
# C5b r=64 final adapter (test loss 0.983)
ADAPTER_C5B = "checkpoints/musicgen_lora_c5_jsb/step_1500"


def chroma_sim(a_path: Path, b_path: Path) -> float:
    import librosa
    a, _ = librosa.load(str(a_path), sr=22050)
    b, _ = librosa.load(str(b_path), sr=22050)
    n = min(len(a), len(b))
    ca = librosa.feature.chroma_cqt(y=a[:n], sr=22050, hop_length=512)
    cb = librosa.feature.chroma_cqt(y=b[:n], sr=22050, hop_length=512)
    nf = min(ca.shape[1], cb.shape[1])
    sims = [
        float(np.dot(ca[:, i], cb[:, i]) /
              (np.linalg.norm(ca[:, i]) * np.linalg.norm(cb[:, i]) + 1e-9))
        for i in range(nf)
    ]
    return float(np.mean(sims))


def main():
    pairs = sorted((p for p in JSB_PAIRS.iterdir() if (p / "melody.wav").exists()),
                    key=lambda p: p.name)
    held_out = pairs[-10:-5]  # 5 chorales from the tail
    print(f"chorales: {[p.name for p in held_out]}")
    prompt = "bach four-part chorale played on church organ"

    rows = []
    for pair in held_out:
        mel = pair / "melody.wav"
        ref = pair / "arrangement.wav"
        base_out = OUT_DIR / f"{pair.name}_base.wav"
        c5_out = OUT_DIR / f"{pair.name}_c5_r32.wav"
        c5b_out = OUT_DIR / f"{pair.name}_c5b_r64.wav"
        print(f"\n=== {pair.name} ===")
        if not base_out.exists():
            print(f"  base...")
            arrange_to_file(str(mel), prompt, str(base_out), duration_s=10.0,
                             model_size="melody-large", seed=0)
        if Path(ADAPTER_C5).exists() and not c5_out.exists():
            print(f"  c5 r=32...")
            arrange_to_file(str(mel), prompt, str(c5_out), duration_s=10.0,
                             model_size="melody-large", seed=0,
                             lora_adapter=ADAPTER_C5)
        if not c5b_out.exists():
            print(f"  c5b r=64...")
            arrange_to_file(str(mel), prompt, str(c5b_out), duration_s=10.0,
                             model_size="melody-large", seed=0,
                             lora_adapter=ADAPTER_C5B)
        base_sim = chroma_sim(mel, base_out)
        c5_sim = chroma_sim(mel, c5_out) if c5_out.exists() else float("nan")
        c5b_sim = chroma_sim(mel, c5b_out)
        ref_sim = chroma_sim(mel, ref)
        row = {"chorale": pair.name, "ref_sim": ref_sim, "base_sim": base_sim,
                "c5_sim": c5_sim, "c5b_sim": c5b_sim}
        rows.append(row)
        print(f"  ref={ref_sim:.3f}  base={base_sim:.3f}  "
              f"c5={c5_sim:.3f}  c5b={c5b_sim:.3f}")

    print("\n=== mean across chorales ===")
    for k in ("ref_sim", "base_sim", "c5_sim", "c5b_sim"):
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        if vals:
            print(f"  {k}: {np.mean(vals):.4f}  (n={len(vals)})")
    out = Path("reports/_phase_f_F4_c5_vs_c5b_multi.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
