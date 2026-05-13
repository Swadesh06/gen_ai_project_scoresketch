"""Phase G G-8: round-trip self-consistency eval.

Run the round-trip distance on every cached ASAP piece and correlate
the distance with each piece's MV2H (from the cached metric file).

Outputs reports/_item-g8.json with per-piece distance and global Pearson
|r| between distance and MV2H.
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.eval.round_trip import round_trip_distance, notes_to_pretty_midi
from humscribe.notes import NoteEvent

CACHE_YMT3 = Path("/workspace/.cache/asap_yourmt3plus")
CACHE_RENDERS = Path("/workspace/.cache/asap_renders")

PIECES = [
    "Bach__Fugue__bwv_846", "Bach__Fugue__bwv_848", "Bach__Fugue__bwv_854",
    "Bach__Fugue__bwv_856", "Bach__Fugue__bwv_857",
    "Beethoven__Piano_Sonatas__21-1", "Schumann__Toccata",
    "Chopin__Berceuse_op_57", "Liszt__Sonata",
]


def _load_ymt3_notes(piece_key: str, eval_seconds: float = 30.0) -> tuple[list[NoteEvent], float]:
    pkl = CACHE_YMT3 / f"{piece_key}.pkl"
    if not pkl.exists():
        return [], 120.0
    with open(pkl, "rb") as f:
        cache = pickle.load(f)
    bpm = float(cache.get("bpm", 120.0))
    notes: list[NoteEvent] = []
    for n in cache["notes"]:
        m = int(n["midi"])
        if 1 <= m <= 127:
            notes.append(NoteEvent(onset_s=float(n["on"]), offset_s=float(n["off"]),
                                    pitch_midi=m, velocity=int(n.get("vel", 80))))
    notes = [x for x in notes if x.onset_s < eval_seconds]
    return notes, bpm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seconds", type=float, default=30.0)
    ap.add_argument("--out", default="reports/_item-g8.json")
    ap.add_argument("--mv2h-source", default="reports/_metric_mv2h_phase_g_asap_g1g2.json",
                    help="json with rows containing per-piece mv2h to correlate against")
    args = ap.parse_args()

    import librosa
    mv2h_data = {}
    if Path(args.mv2h_source).exists():
        mv2h_obj = json.loads(Path(args.mv2h_source).read_text())
        for r in (mv2h_obj.get("asap") or mv2h_obj).get("rows", []):
            mv2h_data[r["piece"]] = r

    rows = []
    t0 = time.time()
    for piece in PIECES:
        wav = CACHE_RENDERS / f"{piece}.wav"
        if not wav.exists():
            continue
        notes, bpm = _load_ymt3_notes(piece, eval_seconds=args.eval_seconds)
        if not notes:
            continue
        audio, sr = librosa.load(str(wav), sr=None, mono=True, duration=args.eval_seconds)
        rt = round_trip_distance(audio, sr, notes, bpm=bpm)
        mv2h = mv2h_data.get(piece, {}).get("mv2h", float("nan"))
        rows.append({"piece": piece, "rt_distance": rt.distance,
                      "mv2h": mv2h, "n_notes": len(notes),
                      "frames_ref": rt.mfcc_ref_frames, "frames_pred": rt.mfcc_pred_frames})
        print(f"{piece:42s} rt_dist={rt.distance:.5f} mv2h={mv2h:.4f} n={len(notes)}")
    pearson = float("nan")
    spearman = float("nan")
    if len(rows) >= 3:
        d = np.array([r["rt_distance"] for r in rows], dtype=np.float64)
        m = np.array([r["mv2h"] for r in rows], dtype=np.float64)
        valid = ~(np.isnan(d) | np.isnan(m))
        if valid.sum() >= 3:
            from scipy.stats import pearsonr, spearmanr
            pearson = float(pearsonr(d[valid], m[valid])[0])
            spearman = float(spearmanr(d[valid], m[valid])[0])
    out = {"rows": rows, "pearson_dist_vs_mv2h": pearson,
           "spearman_dist_vs_mv2h": spearman,
           "wall_s": time.time() - t0,
           "eval_seconds": args.eval_seconds,
           "mv2h_source": args.mv2h_source}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    print(f"pearson(dist, mv2h) = {pearson:+.3f} (|r| = {abs(pearson):.3f})")


if __name__ == "__main__":
    main()
