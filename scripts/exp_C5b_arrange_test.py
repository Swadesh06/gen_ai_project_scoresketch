"""Phase F-4 prep: render a test arrangement with the C5b r=64 adapter.

Generate (base, C5b r=64) arrangements from a held-out JSB chorale melody.
Compare via:
- audio file size + duration sanity
- chroma similarity between melody input and generated output

If C5b output is recognizable (chroma similarity ~0.3-0.7) but distinct
from melody (not just flute-echo), the adapter is doing useful work.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.arrange.musicgen import arrange_to_file


JSB_PAIRS = Path("/workspace/datasets/jsb_pairs")
OUT_DIR = Path("outputs/c5b_arrange_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def chroma_similarity(a_path: Path, b_path: Path) -> float:
    import librosa
    a, sr_a = librosa.load(str(a_path), sr=22050)
    b, sr_b = librosa.load(str(b_path), sr=22050)
    n = min(len(a), len(b))
    ca = librosa.feature.chroma_cqt(y=a[:n], sr=22050, hop_length=512)
    cb = librosa.feature.chroma_cqt(y=b[:n], sr=22050, hop_length=512)
    nf = min(ca.shape[1], cb.shape[1])
    return float(np.mean([np.dot(ca[:, i], cb[:, i]) / (np.linalg.norm(ca[:, i]) * np.linalg.norm(cb[:, i]) + 1e-9) for i in range(nf)]))


def main():
    # Pick a held-out chorale (one we know exists). Use a Bach BWV that
    # wasn't in the C5 training set's first 315 of 371 — pick one of the
    # last alphabetically, e.g. bwv_99.6 or similar.
    pairs = sorted([d for d in JSB_PAIRS.iterdir() if (d / "melody.wav").exists()],
                    key=lambda p: p.name)
    if len(pairs) < 350:
        print(f"only {len(pairs)} pairs; expected 371")
    test_pair = pairs[-10]  # 10 from the end of sorted list
    print(f"using test pair: {test_pair.name}")
    mel = test_pair / "melody.wav"
    ref_arr = test_pair / "arrangement.wav"

    # Generate (base, C5b r=64) variants.
    prompt = "bach four-part chorale played on church organ"
    print(f"prompt: {prompt!r}")

    # Base (no LoRA)
    base_out = OUT_DIR / f"{test_pair.name}_base.wav"
    print("rendering base (no LoRA)...")
    arrange_to_file(str(mel), prompt, str(base_out), duration_s=10.0,
                     model_size="melody-large", seed=0)

    # C5b r=64
    c5b_out = OUT_DIR / f"{test_pair.name}_c5b_r64.wav"
    print("rendering C5b r=64 LoRA...")
    arrange_to_file(str(mel), prompt, str(c5b_out), duration_s=10.0,
                     model_size="melody-large", seed=0,
                     lora_adapter="checkpoints/musicgen_lora_c5_jsb/step_1500")

    # Chroma similarities
    base_sim = chroma_similarity(mel, base_out)
    c5b_sim = chroma_similarity(mel, c5b_out)
    ref_sim = chroma_similarity(mel, ref_arr)
    print(f"\nchroma similarity (melody vs ...):")
    print(f"  ref arrangement (GT JSB four-voice organ): {ref_sim:.3f}")
    print(f"  base MusicGen (no LoRA):                  {base_sim:.3f}")
    print(f"  C5b r=64 LoRA:                            {c5b_sim:.3f}")

    print(f"\noutputs:")
    print(f"  melody:   {mel}")
    print(f"  ref:      {ref_arr}")
    print(f"  base:     {base_out}")
    print(f"  C5b r=64: {c5b_out}")


if __name__ == "__main__":
    main()
