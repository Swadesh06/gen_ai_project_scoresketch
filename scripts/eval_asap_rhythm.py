"""Validate Stage 4 (beat tracking) and Stage 5 (rhythm quantization) on ASAP.
Stage 4 gate: beat F-measure > 0.90 on Bach BWV 846.
Stage 5 gate: >= 90% of notes match the score's quarterLength when given correct beats.
"""
import argparse, json
from pathlib import Path
import mir_eval, music21, numpy as np
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.instrument.piano import transcribe_piano

def main(asap_dir, piece_pattern):
    asap = Path(asap_dir).expanduser()
    ann = json.loads((asap / "asap_annotations.json").read_text())
    # find a matching performance
    perf_keys = [k for k in ann if piece_pattern in k]
    if not perf_keys:
        print(f"No piece matching '{piece_pattern}'"); return
    perf_key = perf_keys[0]
    perf = ann[perf_key]
    audio_path = asap / perf_key.replace(".mid", ".wav")  # if audio rendered
    score_xml  = asap / perf_key.rsplit("/", 1)[0] / "xml_score.musicxml"

    # Stage 4: beat tracking vs ASAP beats
    pred_beats, _, _ = track_beats_beat_this(str(audio_path))
    gt_beats = np.array(perf["performance_beats"])
    f_beat = mir_eval.beat.f_measure(gt_beats, pred_beats, f_measure_threshold=0.07)
    print(f"Stage 4 beat F-measure: {f_beat:.3f}  (gate: > 0.90)")

    # Stage 5: DP quantizer given ASAP beats + ByteDance notes
    notes = transcribe_piano(str(audio_path))   # returns NoteEvents
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, gt_beats)
    pred_durations_quarters = (q_off - q_on) / 12.0  # tatums to quarters

    # ground truth quarterLengths from the MusicXML
    score = music21.converter.parse(str(score_xml))
    gt_durations = [n.quarterLength for n in score.flatten().notes]

    n_match = sum(1 for p, g in zip(pred_durations_quarters, gt_durations)
                  if abs(p - g) < 0.05)
    pct = n_match / max(len(gt_durations), 1)
    print(f"Stage 5 quarterLength match: {pct*100:.1f}%  (gate: > 90%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap-dir", default="~/datasets/asap")
    ap.add_argument("--piece-pattern", default="Bach/Fugue/bwv_846")
    main(**vars(ap.parse_args()))
