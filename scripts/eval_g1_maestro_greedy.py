"""Phase G G-1 strict measurement on MAESTRO with greedy fallback.

The B76 transformer voice tracker was trained on piano left/right-hand
supervision. On MAESTRO chamber clips the GT has 3-4 instrument tracks
and B76's 2-voice output crashes the MV2H voice sub-score.

Strict fix: when the GT side has >2 voices (i.e. input_kind="piano" but
the audio is chamber), bypass B76 and use the greedy multi-voice tracker
(which can produce >=2 voices). Compare:
  baseline (G-1 OFF):  voice = [0]*n
  G-1 B76:             B76 2-voice partition
  G-1 greedy:          greedy adaptive_pj voice partition

If greedy gets MAESTRO voice >= 0.65 (was 0.46), G-1 strict-passes on MAESTRO.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.config import PipelineConfig
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.eval.voice_emission import (
    voice_ids_b76, voice_ids_greedy,
)
from humscribe.notes import NoteEvent
from humscribe.pipeline import transcribe

CLIPS = Path("outputs/maestro_clips")
OUT = Path("reports/_item-g1_maestro_strict.json")


def _gt_text_with_beats(mid: Path, eval_s: float = 30.0) -> tuple[str, dict]:
    pm = pretty_midi.PrettyMIDI(str(mid))
    bpm = float(pm.estimate_tempo()) if pm.instruments else 120.0
    ts_changes = pm.time_signature_changes
    ts = f"{ts_changes[0].numerator}/{ts_changes[0].denominator}" if ts_changes else "4/4"
    notes: list[NoteEvent] = []
    voices: list[int] = []
    for ti, inst in enumerate(pm.instruments):
        for n in inst.notes:
            if n.start >= eval_s:
                continue
            notes.append(NoteEvent(onset_s=float(n.start), offset_s=float(n.end),
                                    pitch_midi=int(n.pitch), velocity=int(n.velocity)))
            voices.append(ti)
    try:
        beats = [float(b) for b in pm.get_beats() if b < eval_s + 5.0]
    except Exception:
        beats = None
    txt = notes_to_mv2h_format(notes, bpm=bpm, time_sig=ts, voices=voices,
                                 tatums_per_beat=4, beats=beats)
    return txt, {"n_notes_gt": len(notes), "n_voices_gt": int(max(voices) + 1) if voices else 0}


def _pred_text(audio: Path, voices_mode: str, eval_s: float = 30.0,
                use_beats: bool = True) -> tuple[str, dict]:
    cfg = PipelineConfig(input_kind="piano", mode="medium", render_svg=False,
                          per_voice_dp="off", transcriber="bytedance_piano",
                          same_pitch_merge="off", median_smooth_g5="off",
                          silent_trim_g6="off", render_tpb_auto="off")
    r = transcribe(str(audio), cfg)
    notes = [n for n in r.notes if n.onset_s < eval_s]
    if voices_mode == "off":
        voices = [0] * len(notes)
    elif voices_mode == "b76":
        v = voice_ids_b76(notes)
        voices = v if v is not None else [0] * len(notes)
    elif voices_mode == "greedy":
        voices = voice_ids_greedy(notes)
    else:
        raise ValueError(voices_mode)
    beats = None
    if use_beats and r.beats is not None and len(r.beats) >= 2:
        beats = [b for b in r.beats if b < eval_s + 5.0]
    txt = notes_to_mv2h_format(
        notes, bpm=float(r.bpm) if r.bpm else 120.0,
        time_sig="4/4", voices=voices, tatums_per_beat=4, beats=beats,
    )
    return txt, {"n_notes_pred": len(notes), "voices_mode": voices_mode,
                  "n_voices_pred": int(max(voices) + 1) if voices else 0}


def main() -> None:
    rows = []
    for wav in sorted(CLIPS.glob("*.wav")):
        mid = wav.with_suffix(".mid")
        if not mid.exists():
            continue
        gt_text, gt_info = _gt_text_with_beats(mid)
        clip_rows: dict[str, dict] = {}
        for mode in ("off", "b76", "greedy"):
            pred_text, pred_info = _pred_text(wav, voices_mode=mode)
            # MAESTRO pred/GT share the same absolute time base (both from the
            # same 30 s audio clip), so 'aligned' is correct and avoids the
            # quadratic DTW blowup on dense chamber polyphony that 'non_aligned'
            # triggers — observed timeout (>300 s) on first clip in initial run.
            try:
                res = compute_mv2h(pred_text, gt_text, align="aligned", timeout_s=120.0)
            except Exception as e:
                print(f"  {mode}: mv2h error: {type(e).__name__}: {e}")
                continue
            clip_rows[mode] = {**res.as_dict(), **pred_info, **gt_info}
        rows.append({"clip": wav.name, **clip_rows})
        for mode in ("off", "b76", "greedy"):
            r = clip_rows[mode]
            print(f"{wav.name[:50]:50s} {mode:6s} mv2h={r['mv2h']:.4f} voice={r['voice']:.3f} mp={r['multi_pitch']:.3f} n_pred_voices={r['n_voices_pred']}")
    # Aggregate
    if rows:
        for mode in ("off", "b76", "greedy"):
            means = {}
            for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony"):
                vals = [r[mode][k] for r in rows]
                means[k] = float(np.mean(vals))
            print(f"\n[mean {mode}]  mv2h={means['mv2h']:.4f}  voice={means['voice']:.3f}  meter={means['meter']:.3f}  mp={means['multi_pitch']:.3f}  value={means['value']:.3f}")
    out = {"per_clip": rows,
            "means": {mode: {k: float(np.mean([r[mode][k] for r in rows]))
                              for k in ("mv2h", "multi_pitch", "voice", "meter", "value", "harmony")}
                       for mode in ("off", "b76", "greedy")} if rows else {}}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
