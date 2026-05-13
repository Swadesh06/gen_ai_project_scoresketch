"""Phase E item 7 ME-11: formant-band onset detector.

Detects onset peaks in the 1.5-3.5 kHz mel-spectrogram band where vocal
formant energy is concentrated and where neural pitch trackers see less
information. Ensemble with the production-direct path: add ME-11 onsets
to direct only when they have no nearby direct counterpart AND a PESTO-
voiced pitch trace at that time.

If this lifts Vocadito A1 noff F1 ≥ 0.69, items 2 and 7 unlock.

Pipeline:
  1. Run production direct on all 40 clips (cached at /workspace/.cache/...)
  2. Run librosa onset_detect on the formant band (1.5-3.5 kHz)
  3. For each ME-11 onset NOT within 50 ms of a direct onset, look up the
     PESTO pitch at that time. If voiced, add it as a note.
  4. Evaluate noff F1 against A1 GT.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC = Path("/home/swadesh/datasets/vocadito")
OUT = Path("reports/_phase_e_item7_me11_formant_onset.json")


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


def _formant_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """Onsets detected on 1.5-3.5 kHz formant band of the mel-spectrogram."""
    n_fft = 2048
    hop = 220  # 10 ms at 22050 Hz
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=64,
        fmin=1500.0, fmax=3500.0, hop_length=hop, n_fft=n_fft,
    )
    log_mel = librosa.power_to_db(mel)
    # Onset envelope on the formant-band log-mel mean.
    env = log_mel.mean(axis=0)
    # Pre-emphasis: rectified positive difference.
    delta = np.maximum(0, np.diff(env, prepend=env[0]))
    # Smooth + peak-pick.
    onset_env = librosa.util.normalize(delta) if delta.max() > 0 else delta
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        backtrack=False, wait=3, pre_avg=3, post_avg=3,
        pre_max=5, post_max=5, delta=0.1,
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)


def main():
    mc = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    rows = []
    t_total = time.time()
    for cid in range(1, 41):
        wav = VOC / "Audio" / f"vocadito_{cid}.wav"
        if not wav.exists(): continue
        gt = _gt_notes(cid)
        if not gt: continue
        y, sr = load_audio(str(wav), target_sr=22050)
        # Direct path
        t, hz, vc = track_pitch_hybrid_voicing(y, sr)
        direct_notes = segment_pitch_to_notes(t, hz, vc, mc)
        direct = [(n.onset_s, n.offset_s, n.midi())
                  for n in direct_notes if n.midi() >= 1]
        # ME-11 formant onsets
        f_onsets = _formant_onsets(y, sr)
        # Vote: add ME-11 onset only if (a) no direct onset within 50ms
        # AND (b) PESTO pitch trace exists at that time AND (c) voicing >= mc.voicing_threshold.
        added = []
        for fo in f_onsets:
            if any(abs(fo - d[0]) < 0.05 for d in direct):
                continue
            # Look up PESTO sample within 30 ms of formant onset
            idx_near = np.argmin(np.abs(t - fo))
            if abs(t[idx_near] - fo) > 0.03: continue
            if vc[idx_near] < mc.voicing_threshold * 0.7: continue
            if hz[idx_near] <= 0: continue
            m = _hz_to_midi(float(hz[idx_near]))
            if m < 1: continue
            # Find next direct onset to bound offset
            next_on = min((d[0] for d in direct if d[0] > fo), default=fo + 0.5)
            off = min(next_on - 0.005, fo + 0.5)
            if off - fo < 0.05: off = fo + 0.05
            added.append((fo, off, m))
        ens = sorted(direct + added, key=lambda n: n[0])
        f1_direct = _noff_f1(direct, gt)
        f1_ens = _noff_f1(ens, gt)
        row = {"cid": cid, "n_gt": len(gt), "n_direct": len(direct),
               "n_me11_added": len(added), "n_ens": len(ens),
               "f1_direct": f1_direct, "f1_ens": f1_ens,
               "delta": f1_ens - f1_direct}
        rows.append(row)
        print(f"voc_{cid:2d}: dir={f1_direct:.3f}  ens={f1_ens:.3f}  Δ={f1_ens-f1_direct:+.3f}  "
              f"(added {len(added)} ME-11 onsets)", flush=True)
    mean_d = float(np.mean([r["f1_direct"] for r in rows]))
    mean_e = float(np.mean([r["f1_ens"] for r in rows]))
    print(f"\n=== 40-clip means ===")
    print(f"  direct:    {mean_d:.4f}")
    print(f"  +ME-11:    {mean_e:.4f}")
    print(f"  delta:     {mean_e - mean_d:+.4f}")
    print(f"  v3 item-2 noff ≥ 0.69:  {'PASS' if mean_e >= 0.69 else f'FAIL ({mean_e:.4f})'}")
    print(f"  v3 item-7 noff ≥ 0.70:  {'PASS' if mean_e >= 0.70 else f'FAIL ({mean_e:.4f})'}")
    OUT.write_text(json.dumps({"rows": rows, "mean_direct": mean_d,
                                "mean_ensemble": mean_e,
                                "delta": mean_e - mean_d}, indent=2))
    print(f"wrote {OUT}  (wall: {(time.time()-t_total)/60:.1f} min)")


if __name__ == "__main__":
    main()
