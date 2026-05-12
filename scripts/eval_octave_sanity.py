"""Validate the octave sanity check on the 9 ASAP pieces.

For each piece compare beat_this output vs GT beats and report whether
the detector correctly flags the octave-misaligned pieces.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretty_midi

from humscribe.beat.octave_sanity import (
    detect_octave_misalignment, apply_octave_correction,
)
from humscribe.notes import NoteEvent

CACHE = Path("/workspace/.cache/sweep_e6_features")
ASAP_REPO = Path("/workspace/.cache/asap_score_midis/asap-repo")

PIECES = {
    "Bach__Fugue__bwv_854": "Bach/Fugue/bwv_854",
    "Bach__Fugue__bwv_846": "Bach/Fugue/bwv_846",
    "Bach__Fugue__bwv_848": "Bach/Fugue/bwv_848",
    "Bach__Fugue__bwv_856": "Bach/Fugue/bwv_856",
    "Bach__Fugue__bwv_857": "Bach/Fugue/bwv_857",
    "Beethoven__Piano_Sonatas__21-1": "Beethoven/Piano_Sonatas/21-1",
    "Chopin__Berceuse_op_57": "Chopin/Berceuse_op_57",
    "Liszt__Sonata": "Liszt/Sonata",
    "Schumann__Toccata": "Schumann/Toccata",
}


def main():
    rows = []
    for key, piece_dir in PIECES.items():
        npz = CACHE / f"asap_{key}.npz"
        gt = ASAP_REPO / piece_dir / "midi_score.mid"
        if not (npz.exists() and gt.exists()): continue
        d = np.load(npz)
        beats = d["beats"]
        notes = []
        for i in range(len(d["notes_on"])):
            m = int(d["notes_midi"][i])
            if m < 1: continue
            notes.append(NoteEvent(onset_s=float(d["notes_on"][i]),
                                    offset_s=float(d["notes_off"][i]),
                                    pitch_midi=m, velocity=80))
        diag = detect_octave_misalignment(beats, notes)
        gt_pm = pretty_midi.PrettyMIDI(str(gt))
        gt_beats = gt_pm.get_beats(start_time=0.0)
        gt_beats = gt_beats[gt_beats < 30.0]
        true_bpm = 60.0 / float(np.median(np.diff(gt_beats))) if len(gt_beats) >= 2 else 120.0
        pred_bpm = 60.0 / diag["median_beat_ioi"]
        ratio_to_true = pred_bpm / max(true_bpm, 1e-3)
        rows.append({"piece": key, "true_bpm": true_bpm,
                      "pred_bpm": pred_bpm,
                      "ratio_pred_over_true": ratio_to_true,
                      "n_beats_pred": diag["n_beats"],
                      "n_beats_gt": int(len(gt_beats)),
                      **diag})
        print(f"{key:42s}  pred_bpm={pred_bpm:6.1f}  true_bpm={true_bpm:6.1f}  "
              f"ratio={ratio_to_true:.2f}  recommend={diag['recommend']}  "
              f"note_ioi={diag['median_note_ioi']:.3f}  nppb={diag['notes_per_beat']:.2f}")

    # Verify diagnoses against truth:
    print("\nDiagnosis correctness:")
    for r in rows:
        truth = "keep"
        if r["ratio_pred_over_true"] > 1.7:
            truth = "halve"
        elif r["ratio_pred_over_true"] < 0.6:
            truth = "double"
        match = "✓" if r["recommend"] == truth else "✗"
        print(f"  {r['piece']:42s}  truth={truth:6s}  pred={r['recommend']:6s}  {match}")

    out = Path("reports/_phase_f_F1_octave_sanity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
