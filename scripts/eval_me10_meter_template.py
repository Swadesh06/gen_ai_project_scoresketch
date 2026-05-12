"""Validate ME-10 meter-template ensemble on the 9 ASAP pieces.

For each piece, run the meter-template detector and compare its choice
to the GT time signature from the score MIDI.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.ensemble.me10_meter_template import best_time_signature
from humscribe.notes import NoteEvent

CACHE = Path("/workspace/.cache/sweep_e6_features")
ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
PIECES = {
    "asap_Bach__Fugue__bwv_854": "Bach/Fugue/bwv_854",
    "asap_Bach__Fugue__bwv_846": "Bach/Fugue/bwv_846",
    "asap_Bach__Fugue__bwv_848": "Bach/Fugue/bwv_848",
    "asap_Bach__Fugue__bwv_856": "Bach/Fugue/bwv_856",
    "asap_Bach__Fugue__bwv_857": "Bach/Fugue/bwv_857",
    "asap_Beethoven__Piano_Sonatas__21-1": "Beethoven/Piano_Sonatas/21-1",
    "asap_Chopin__Berceuse_op_57": "Chopin/Berceuse_op_57",
    "asap_Liszt__Sonata": "Liszt/Sonata",
    "asap_Schumann__Toccata": "Schumann/Toccata",
}


def _gt_time_sig(piece_dir: str) -> tuple[int, int] | None:
    midi = ASAP_REPO / piece_dir / "midi_score.mid"
    if not midi.exists():
        return None
    pm = pretty_midi.PrettyMIDI(str(midi))
    if not pm.time_signature_changes:
        return None
    ts = pm.time_signature_changes[0]
    return (ts.numerator, ts.denominator)


def main():
    rows = []
    for k, piece_dir in PIECES.items():
        npz = CACHE / f"{k}.npz"
        if not npz.exists(): continue
        d = np.load(npz)
        notes = []
        for i in range(len(d["notes_on"])):
            m = int(d["notes_midi"][i])
            if m < 1: continue
            notes.append(NoteEvent(onset_s=float(d["notes_on"][i]),
                                    offset_s=float(d["notes_off"][i]),
                                    pitch_midi=m, velocity=80))
        gt = _gt_time_sig(piece_dir)
        chosen = best_time_signature(notes, d["beats"])
        match = "✓" if gt is not None and chosen.time_sig == gt else "✗"
        rows.append({"piece": k,
                      "chosen": f"{chosen.time_sig[0]}/{chosen.time_sig[1]}",
                      "gt": f"{gt[0]}/{gt[1]}" if gt else "?",
                      "score": chosen.score,
                      "n_notes": chosen.n_notes,
                      "match": (chosen.time_sig == gt) if gt else None})
        print(f"{k:50s} chosen={chosen.time_sig[0]}/{chosen.time_sig[1]} "
              f"gt={gt[0] if gt else '?'}/{gt[1] if gt else '?'} {match}")
    n_match = sum(1 for r in rows if r["match"] is True)
    print(f"\nmatched: {n_match}/{len(rows)}")
    out = Path("reports/_exp_ME10_meter_template.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "n_match": n_match,
                                "n_total": len(rows)}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
