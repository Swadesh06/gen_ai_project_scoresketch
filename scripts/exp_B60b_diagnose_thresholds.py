"""B60b: print median IOI and median duration for ByteDance output on each ASAP piece.
Find a threshold that uniquely picks Chopin (where bp wins)."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

ASAP = Path("~/datasets/asap").expanduser()
CACHE_DIR = Path("/workspace/.cache/asap_bytedance")

PIECES = ["Bach/Fugue/bwv_846", "Beethoven/Piano_Sonatas/21-1",
          "Schumann/Toccata", "Chopin/Berceuse_op_57"]


def main():
    print(f"  {'piece':36s}  med_ioi  med_dur  notes/s  total_dur")
    print("  " + "-" * 80)
    for piece_rel in PIECES:
        cache = CACHE_DIR / (piece_rel.replace("/", "__") + ".pkl")
        if not cache.exists(): continue
        with open(cache, "rb") as f: d = pickle.load(f)
        notes = d["notes"]
        if not notes: continue
        onsets = sorted(n["on"] for n in notes)
        iois = np.diff(onsets)
        durs = [n["off"] - n["on"] for n in notes]
        total = onsets[-1]
        nps = len(notes) / total if total > 0 else 0
        print(f"  {piece_rel:36s}  {np.median(iois):7.3f}  {np.median(durs):7.3f}  {nps:7.2f}  {total:7.1f}s")


if __name__ == "__main__":
    main()
