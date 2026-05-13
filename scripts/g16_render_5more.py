"""G-16: render the 5 additional C5b pair clips to bring the listening
test artifact from 5/10 to 10/10. Mirrors `scripts/exp_C5_vs_C5b_multi.py`
but only renders base + c5b_r64 (skipping c5_r32 since the listening test
compares base vs c5b only) for 5 chorales not yet in the artifact set.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.arrange.musicgen import arrange_to_file


JSB_PAIRS = Path("/workspace/datasets/jsb_pairs")
OUT_DIR = Path("outputs/c5_vs_c5b_multi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADAPTER_C5B = "checkpoints/musicgen_lora_c5_jsb/step_1500"

# 5 additional chorales (avoid duplicates with the existing 85.6 / 86.6 / 87.7 / 88.7 / 89.6).
EXTRA = ["bwv90.5", "bwv91.6", "bwv94.8", "bwv96.6", "bwv101.7"]


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


def main() -> None:
    prompt = "bach four-part chorale played on church organ"
    rows = []
    for name in EXTRA:
        pair = JSB_PAIRS / name
        if not (pair / "melody.wav").exists():
            print(f"skip {name}: melody.wav missing")
            continue
        mel = pair / "melody.wav"
        ref = pair / "arrangement.wav"
        base_out = OUT_DIR / f"{name}_base.wav"
        c5b_out = OUT_DIR / f"{name}_c5b_r64.wav"
        if not base_out.exists():
            print(f"=== {name} base ===")
            arrange_to_file(str(mel), prompt, str(base_out), duration_s=10.0,
                             model_size="melody-large", seed=0)
        if not c5b_out.exists():
            print(f"=== {name} c5b r=64 ===")
            arrange_to_file(str(mel), prompt, str(c5b_out), duration_s=10.0,
                             model_size="melody-large", seed=0,
                             lora_adapter=ADAPTER_C5B)
        base_sim = chroma_sim(mel, base_out)
        c5b_sim = chroma_sim(mel, c5b_out)
        ref_sim = chroma_sim(mel, ref)
        rows.append({"chorale": name, "ref_sim": ref_sim,
                      "base_sim": base_sim, "c5b_sim": c5b_sim})
        print(f"{name:10s}  ref={ref_sim:.3f}  base={base_sim:.3f}  c5b={c5b_sim:.3f}")
    out = Path("reports/_item-g16_extra_pairs.json")
    out.write_text(json.dumps({"rows": rows, "n_extra": len(rows)}, indent=2))
    print(f"wrote {out}; total pair set now {5 + len(rows)} / 10")


if __name__ == "__main__":
    main()
