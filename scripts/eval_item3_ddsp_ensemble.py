"""Phase E item 3 strict eval: DDSP humming→violin→re-transcribe + ensemble.

Three configurations measured per Vocadito clip:
- direct:   humming pipeline on raw hum audio (current production baseline)
- ddsp:     humming pipeline on DDSP-violin-transferred audio
- ensemble: per-note majority vote of (direct, ddsp) onsets

v3 pass criteria:
- direct: Voc A1 noff F1 ≥ 0.55 (acceptable but probably worse than 0.665)
- ensemble: Voc A1 noff F1 ≥ 0.71 (the win)

Cached transferred audio at /workspace/.cache/ddsp_violin_vocadito/<id>.wav
so re-runs are fast.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig, ModeConfig
from humscribe.notes import NoteEvent
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC = Path("/home/swadesh/datasets/vocadito")
CACHE = Path("/workspace/.cache/ddsp_violin_vocadito")
CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path("reports/_phase_e_item3_ddsp_ensemble.json")


def _hz_to_midi(hz: float) -> int:
    if hz <= 0: return -1
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _gt_notes(cid: int):
    csv = VOC / "Annotations" / "Notes" / f"vocadito_{cid}_notesA1.csv"
    out = []
    if not csv.exists(): return out
    for line in csv.read_text().splitlines():
        if not line.strip(): continue
        a, b, c = line.split(",")
        on, freq, dur = float(a), float(b), float(c)
        m = _hz_to_midi(freq)
        if m >= 1:
            out.append((on, on + dur, m))
    return out


def _noff_f1(pred, gt) -> float:
    """No-offset F1: onset within 50ms + same pitch class. No offset check."""
    if not pred or not gt: return 0.0
    mg, mp = set(), set()
    for i, (po, pf, pm) in enumerate(pred):
        for j, (go, gf, gm) in enumerate(gt):
            if j in mg: continue
            if abs(po - go) > 0.05: continue
            if pm != gm: continue
            mg.add(j); mp.add(i); break
    tps = len(mp)
    p = tps / max(len(pred), 1)
    r = tps / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-6)


def _violin_cached(cid: int, raw_audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    cache_path = CACHE / f"vocadito_{cid}.wav"
    if cache_path.exists():
        y, sr_out = sf.read(str(cache_path))
        return np.asarray(y, dtype=np.float32), int(sr_out)
    from humscribe.pitch.timbre_transfer.ddsp_violin import transfer
    print(f"  DDSP transfer vocadito_{cid} ({len(raw_audio)/sr:.1f}s)...", flush=True)
    t0 = time.time()
    violin, vsr = transfer(raw_audio, sr)
    print(f"    done in {time.time()-t0:.1f}s ({len(violin)/vsr:.1f}s audio)", flush=True)
    sf.write(str(cache_path), violin.astype(np.float32), vsr)
    return violin.astype(np.float32), vsr


def _transcribe(audio: np.ndarray, sr: int) -> list[tuple[float, float, int]]:
    """Run the humming pipeline pieces (PESTO + CREPE voicing + segmenter)."""
    mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    t, hz, vc = track_pitch_hybrid_voicing(audio, sr)
    notes = segment_pitch_to_notes(t, hz, vc, mc)
    return [(n.onset_s, n.offset_s, n.midi()) for n in notes if n.midi() >= 1]


def _ensemble(direct: list, ddsp: list, gt: list) -> list:
    """Majority-vote-like ensemble: keep direct as base; merge in DDSP notes
    that have NO direct neighbour within 50 ms. Sorted by onset.

    Rationale: the DDSP path lifts hard-to-detect onsets (chest-voice or
    breathy notes where vocal pitch tracker fails) without throwing away
    direct's high-precision base.
    """
    out = list(direct)
    used = set()
    for d in direct:
        used.add(round(d[0] * 100))  # 10 ms bucket
    for d in ddsp:
        bucket = round(d[0] * 100)
        if any(abs(d[0] - x[0]) < 0.05 and x[2] == d[2] for x in direct):
            continue
        out.append(d)
    out.sort(key=lambda n: n[0])
    return out


def main():
    rows = []
    t_total = time.time()
    for cid in range(1, 41):
        wav = VOC / "Audio" / f"vocadito_{cid}.wav"
        if not wav.exists(): continue
        gt = _gt_notes(cid)
        if not gt:
            print(f"voc_{cid:2d}: no GT — skip"); continue
        # Direct path: 22050 Hz, our heuristic humming pipeline.
        y22, sr22 = load_audio(str(wav), target_sr=22050)
        direct = _transcribe(y22, sr22)
        # DDSP path: transfer at 16k, then transcribe at 22050 (resample).
        y16, sr16 = load_audio(str(wav), target_sr=16000)
        violin, vsr = _violin_cached(cid, y16, sr16)
        # Resample violin to 22050 for the humming pipeline.
        import librosa
        violin22 = librosa.resample(violin.astype(np.float32),
                                    orig_sr=vsr, target_sr=22050)
        ddsp_notes = _transcribe(violin22, 22050)
        # Ensemble
        ens = _ensemble(direct, ddsp_notes, gt)
        # Eval
        f1_direct = _noff_f1(direct, gt)
        f1_ddsp = _noff_f1(ddsp_notes, gt)
        f1_ens = _noff_f1(ens, gt)
        row = {"cid": cid, "n_gt": len(gt),
               "n_direct": len(direct), "n_ddsp": len(ddsp_notes),
               "n_ens": len(ens),
               "f1_direct": f1_direct, "f1_ddsp": f1_ddsp,
               "f1_ensemble": f1_ens}
        rows.append(row)
        print(f"voc_{cid:2d}: f1_direct={f1_direct:.3f}  f1_ddsp={f1_ddsp:.3f}  "
              f"f1_ens={f1_ens:.3f}  (gt={len(gt)} dir={len(direct)} "
              f"ddsp={len(ddsp_notes)} ens={len(ens)})", flush=True)
        # Periodic save
        if cid % 5 == 0:
            _save(rows)
    _save(rows)
    print(f"\ntotal wall: {(time.time()-t_total)/60:.1f} min")
    print(f"=== mean F1 across {len(rows)} clips ===")
    for k in ("f1_direct", "f1_ddsp", "f1_ensemble"):
        v = float(np.mean([r[k] for r in rows]))
        print(f"  {k}: {v:.4f}")


def _save(rows):
    out = {"rows": rows}
    if rows:
        for k in ("f1_direct", "f1_ddsp", "f1_ensemble"):
            out[f"mean_{k}"] = float(np.mean([r[k] for r in rows]))
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
