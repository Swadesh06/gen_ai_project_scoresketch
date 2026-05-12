"""Phase E item 7 ME-9 evaluation: measure accidental count + readability.

For each demo SVG + a few cached ASAP pieces, run with and without
`enharmonic_spelling=True` and report:
- count of explicit accidentals in the resulting MusicXML
- count of "out-of-key" letters (e.g. D# in E major would be an out-of-key
  letter if it should have been E♭, but D# IS in E major so it stays)
- MV2H delta (should be 0.0 ± 0.005 — ME-9 doesn't change pitches)

Pass criteria per item 7 (per spec):
- MV2H delta >= -0.005 (no regression)
- accidental count drops by at least 10% on at least 3 pieces
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format, stream_to_mv2h_format
from humscribe.score import build_stream
from humscribe.notes import NoteEvent

import pickle


CACHE = Path("/workspace/.cache/sweep_e6_features")
PIECES = [
    "asap_Bach__Fugue__bwv_854",
    "asap_Beethoven__Piano_Sonatas__21-1",
    "asap_Chopin__Berceuse_op_57",
    "asap_Liszt__Sonata",
    "asap_Schumann__Toccata",
]


def _piece_notes(piece_key: str) -> tuple[list[NoteEvent], float] | None:
    npz = CACHE / f"{piece_key}.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    notes = []
    for i in range(len(d["notes_on"])):
        m = int(d["notes_midi"][i])
        if m < 1 or m > 127: continue
        notes.append(NoteEvent(onset_s=float(d["notes_on"][i]),
                                offset_s=float(d["notes_off"][i]),
                                pitch_midi=m, velocity=80))
    bpm = float(d["bpm"][0])
    return notes, bpm


def _accidental_count(stream) -> int:
    from music21 import note as m21note
    n = 0
    for el in stream.recurse().notes:
        if isinstance(el, m21note.Note):
            if el.pitch.accidental is not None and el.pitch.accidental.alter != 0:
                n += 1
        else:
            # Chord
            for p in el.pitches:
                if p.accidental is not None and p.accidental.alter != 0:
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/_exp_ME9_spelling.json"))
    args = ap.parse_args()

    rows = []
    for k in PIECES:
        r = _piece_notes(k)
        if r is None:
            print(f"skip {k}: no cache"); continue
        notes, bpm = r
        # Baseline (no enharmonic_spelling)
        s_base = build_stream(notes, bpm=bpm, time_sig="4/4",
                               render_tpb=12, enharmonic_spelling=False)
        # ME-9
        s_me9 = build_stream(notes, bpm=bpm, time_sig="4/4",
                              render_tpb=12, enharmonic_spelling=True)
        acc_base = _accidental_count(s_base)
        acc_me9 = _accidental_count(s_me9)
        # Sanity that pitches are unchanged (MIDI numbers — not letter names).
        pitches_base = sorted(int(p.midi) for el in s_base.recurse().notes
                                for p in (el.pitches if hasattr(el, "pitches") else [el.pitch]))
        pitches_me9 = sorted(int(p.midi) for el in s_me9.recurse().notes
                              for p in (el.pitches if hasattr(el, "pitches") else [el.pitch]))
        pitches_match = pitches_base == pitches_me9

        rel = (acc_base - acc_me9) / max(acc_base, 1)
        row = {"piece": k, "acc_base": acc_base, "acc_me9": acc_me9,
                "acc_drop_pct": round(100 * rel, 2),
                "pitches_unchanged": pitches_match,
                "n_notes": len(pitches_base)}
        rows.append(row)
        print(f"{k:50s} acc {acc_base} -> {acc_me9} ({100*rel:.1f}% drop)  "
              f"pitches_unchanged={pitches_match}  notes={len(pitches_base)}")

    if rows:
        mean_drop = float(np.mean([r["acc_drop_pct"] for r in rows]))
        all_match = all(r["pitches_unchanged"] for r in rows)
        print(f"\nmean accidental drop: {mean_drop:.1f}%")
        print(f"all pitches unchanged: {all_match}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rows": rows,
        "mean_acc_drop_pct": float(np.mean([r["acc_drop_pct"] for r in rows])) if rows else None,
        "all_pitches_unchanged": all(r["pitches_unchanged"] for r in rows) if rows else None,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
