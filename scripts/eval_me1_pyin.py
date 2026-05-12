"""ME-1 pYIN diversifier — evaluate ensemble vs PESTO+CREPE on 10 Vocadito clips.

For each clip, compare:
- baseline: PESTO + CREPE-voicing (current production)
- ME-1 ensemble: PESTO + CREPE-voicing + pYIN agreement-vote on voicing
The metric is MV2H against A1 annotations (post-DP-quantized for both).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.config import ModeConfig
from humscribe.ensemble.me1_pyin import track_pitch_pyin, vote_with_pesto_crepe
from humscribe.eval.mv2h import compute_mv2h
from humscribe.eval.mv2h_io import notes_to_mv2h_format
from humscribe.notes import NoteEvent
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm

VOC_AUDIO = Path("/home/swadesh/datasets/vocadito/Audio")
VOC_NOTES = Path("/home/swadesh/datasets/vocadito/Annotations/Notes")
CACHE = Path("/workspace/.cache/sweep_e6_features")


def _gt_text(clip_id: str, annotator: str = "A1") -> tuple[str, int] | None:
    gt_path = CACHE / f"voc_{clip_id}_{annotator}_gt.txt"
    if gt_path.exists():
        # n_notes can be inferred from file
        txt = gt_path.read_text()
        n = txt.count("Note ")
        return txt, n
    return None


def _baseline_notes(clip_id: int) -> tuple[list[NoteEvent], np.ndarray, float] | None:
    audio = VOC_AUDIO / f"vocadito_{clip_id}.wav"
    if not audio.exists(): return None
    y, sr = load_audio(str(audio), target_sr=22050)
    t, hz, vc = track_pitch_hybrid_voicing(y, sr)
    mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    notes = segment_pitch_to_notes(t, hz, vc, mc)
    beats, _, bpm = track_beats_beat_this(str(audio), target_bpm=110.0)
    return notes, beats, bpm, t, hz, vc, y, sr


def _me1_notes(t_p, hz_p, vc_p, y, sr) -> list[NoteEvent]:
    """Apply pYIN agreement-vote on the PESTO+CREPE voicing signal."""
    t_pyin, hz_pyin, vc_pyin = track_pitch_pyin(y, sr)
    t_e, hz_e, vc_e = vote_with_pesto_crepe(t_p, hz_p, vc_p,
                                              t_pyin, hz_pyin, vc_pyin)
    mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    return segment_pitch_to_notes(t_e, hz_e, vc_e, mc)


def _emit_with_dp(notes: list[NoteEvent], beats: np.ndarray, bpm: float,
                  tpb: int = 24) -> str:
    on = np.array([n.onset_s for n in notes], dtype=np.float64)
    off = np.array([n.offset_s for n in notes], dtype=np.float64)
    if len(on) > 0 and len(beats) >= 2:
        q_on, q_off = viterbi_quantize_rhythm(on, off, beats,
                                                tatums_per_beat=tpb,
                                                offgrid_penalty=0.5)
        tatum_s = 60.0 / (max(bpm, 1e-3) * tpb)
        on_origin = float(on[0]) - q_on[0] * tatum_s
        new = []
        for i, n in enumerate(notes):
            new_on = on_origin + int(q_on[i]) * tatum_s
            new_off = on_origin + int(q_off[i]) * tatum_s
            if new_off <= new_on: new_off = new_on + tatum_s
            new.append(NoteEvent(onset_s=new_on, offset_s=new_off,
                                  pitch_midi=n.midi(), velocity=n.velocity))
        notes = new
    return notes_to_mv2h_format(notes, bpm=bpm if bpm > 0 else 120.0,
                                time_sig="4/4", voices=[0]*len(notes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/_exp_ME1_pyin.json"))
    args = ap.parse_args()
    rows = []
    for cid in range(1, args.limit + 1):
        gt = _gt_text(str(cid), "A1")
        if gt is None:
            print(f"skip voc_{cid}: no GT cache"); continue
        gt_text, n_gt = gt
        b = _baseline_notes(cid)
        if b is None:
            print(f"skip voc_{cid}: no audio"); continue
        notes_b, beats, bpm, t_p, hz_p, vc_p, y, sr = b
        notes_m1 = _me1_notes(t_p, hz_p, vc_p, y, sr)
        try:
            pred_b = _emit_with_dp(notes_b, beats, bpm)
            pred_m1 = _emit_with_dp(notes_m1, beats, bpm)
            mv_b = compute_mv2h(pred_b, gt_text, align="non_aligned",
                                  timeout_s=60.0)
            mv_m1 = compute_mv2h(pred_m1, gt_text, align="non_aligned",
                                   timeout_s=60.0)
        except Exception as e:
            print(f"voc_{cid} mv2h err: {e}"); continue
        delta = mv_m1.mv2h - mv_b.mv2h
        row = {"clip_id": cid, "n_base": len(notes_b), "n_me1": len(notes_m1),
                "mv2h_base": mv_b.mv2h, "mv2h_me1": mv_m1.mv2h, "delta": delta}
        rows.append(row)
        print(f"voc_{cid:2d}  n: {len(notes_b)}→{len(notes_m1)}  "
              f"mv2h: {mv_b.mv2h:.4f}→{mv_m1.mv2h:.4f}  Δ={delta:+.4f}")
    if rows:
        mean_b = float(np.mean([r["mv2h_base"] for r in rows]))
        mean_m1 = float(np.mean([r["mv2h_me1"] for r in rows]))
        print(f"\nmean baseline = {mean_b:.4f}")
        print(f"mean ME-1     = {mean_m1:.4f}")
        print(f"mean delta    = {mean_m1-mean_b:+.4f}")
    args.out.write_text(json.dumps({"rows": rows,
                                      "mean_base": float(np.mean([r["mv2h_base"] for r in rows])) if rows else None,
                                      "mean_me1": float(np.mean([r["mv2h_me1"] for r in rows])) if rows else None,
                                      "mean_delta": float(np.mean([r["delta"] for r in rows])) if rows else None,
                                     }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
