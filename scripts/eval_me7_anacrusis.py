"""Phase E item 7 ME-7 eval: anacrusis (pickup-note) detection.

For each cached ASAP piece, run the anacrusis detector and report whether
the first note is flagged. Cross-check against GT MIDI to see if the first
note in the score is on beat 1 (no pickup) or before beat 1 (pickup).

Goal: validate that the detector is correct on ground-truth, then we can
wire it into the pipeline to shift beats when it fires.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.ensemble.me7_anacrusis import detect_anacrusis
from humscribe.notes import NoteEvent

CACHE = Path("/workspace/.cache/sweep_e6_features")
ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")
PIECES = {
    "asap_Bach__Fugue__bwv_854": "Bach/Fugue/bwv_854",
    "asap_Beethoven__Piano_Sonatas__21-1": "Beethoven/Piano_Sonatas/21-1",
    "asap_Chopin__Berceuse_op_57": "Chopin/Berceuse_op_57",
    "asap_Liszt__Sonata": "Liszt/Sonata",
    "asap_Schumann__Toccata": "Schumann/Toccata",
}


def _gt_first_note_offset_beats(piece_dir: str) -> float:
    """Return the first note's onset in beats. If <0.95 it's likely a pickup."""
    mid = ASAP_REPO / piece_dir / "midi_score.mid"
    if not mid.exists():
        return float("nan")
    pm = pretty_midi.PrettyMIDI(str(mid))
    if not pm.instruments:
        return float("nan")
    first_note_t = min(n.start for inst in pm.instruments for n in inst.notes)
    # Estimate beat from first time signature + tempo
    bpm = float(pm.estimate_tempo()) if pm.instruments else 120.0
    return first_note_t * (bpm / 60.0)


def main():
    rows = []
    for k, gt_dir in PIECES.items():
        npz = CACHE / f"{k}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        on = d["notes_on"]; off = d["notes_off"]; midi = d["notes_midi"]
        beats = d["beats"]
        notes = []
        for i in range(min(20, len(on))):
            m = int(midi[i])
            if m < 1: continue
            notes.append(NoteEvent(onset_s=float(on[i]), offset_s=float(off[i]),
                                   pitch_midi=m, velocity=80))
        res = detect_anacrusis(notes, beats=beats)
        gt_beats = _gt_first_note_offset_beats(gt_dir)
        rows.append({"piece": k, "detected_pickup": res.is_pickup,
                      "pickup_dur": res.pickup_duration_s,
                      "next4_mean": res.next_4_mean_duration_s,
                      "reason": res.reason,
                      "gt_first_note_beats": gt_beats})
        print(f"{k:50s} pickup={res.is_pickup}  reason={res.reason[:50]}  "
              f"gt_first_beat={gt_beats:.2f}")
    out = Path("reports/_exp_ME7_anacrusis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
