"""B47: voicing hysteresis — on-threshold and off-threshold differ.
Onset triggers at high vt; note continues until voicing drops below low vt.
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, hz_to_midi, midi_to_hz
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def median_filter(x: np.ndarray, w: int) -> np.ndarray:
    w = max(int(w) | 1, 1)
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + w])
    return out


def segment_hysteresis(times, hz, voicing, vt_on, vt_off, psw, mns, oms):
    if len(times) == 0: return []
    midi = np.where(hz > 0, np.array([hz_to_midi(float(h)) for h in hz]), 0.0)
    smooth = median_filter(midi, psw)
    n = len(times)
    states = np.zeros(n, dtype=bool)
    in_note = False
    for i in range(n):
        if not in_note and voicing[i] >= vt_on:
            in_note = True
        elif in_note and voicing[i] < vt_off:
            in_note = False
        states[i] = in_note
    # Find contiguous runs
    segs = []
    i = 0
    while i < n:
        if not states[i]:
            i += 1; continue
        j = i
        while j + 1 < n and states[j + 1]: j += 1
        segs.append((i, j)); i = j + 1
    # Merge close segments
    if not segs: return []
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if times[s] - times[pe] < oms:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    notes = []
    for s, e in merged:
        sub_t = times[s:e+1]; sub_m = smooth[s:e+1]; sub_v = voicing[s:e+1]
        # Split on pitch change
        start = 0
        cur_med = float(np.median(sub_m[:max(int(len(sub_m) * 0.2), 1)]))
        for k in range(1, len(sub_t)):
            if abs(float(sub_m[k]) - cur_med) > 0.5:
                _make_note(sub_t, sub_m, sub_v, start, k - 1, mns, notes)
                start = k
                cur_med = float(sub_m[k])
        _make_note(sub_t, sub_m, sub_v, start, len(sub_t) - 1, mns, notes)
    return notes


def _make_note(t, m, v, s, e, mns, out):
    if e <= s: return
    midi_med = float(np.median(m[s:e+1]))
    midi_int = int(round(midi_med)) if midi_med > 0 else 0
    if midi_int <= 0: return
    on_t = float(t[s])
    off_t = float(t[e]) + (float(t[1] - t[0]) if len(t) > 1 else 0.01)
    if (off_t - on_t) < mns: return
    out.append(NoteEvent(onset_s=on_t, offset_s=off_t,
                         pitch_hz=midi_to_hz(midi_med), pitch_midi=midi_int,
                         confidence=float(np.mean(v[s:e+1]))))


def predict(wav: Path, vt_on, vt_off, psw, mns, oms):
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, _pv = track_pitch_pesto(audio, sr)
    ct, _ch, cv = track_pitch_crepe(audio, sr)
    voicing = np.interp(pt, ct, cv)
    return segment_hysteresis(pt, ph, voicing, vt_on, vt_off, psw, mns, oms)


def score(notes, iv, hz):
    if not notes: return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, hz, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B47_hysteresis",
                     config={"git_sha": git_sha()}, tags=["B47", "vocadito", "hysteresis"], dir="logs/wandb")
    rows = []
    for vt_on in (0.75, 0.80, 0.85):
        for vt_off in (0.30, 0.40, 0.50, 0.60, 0.70):
            if vt_off >= vt_on: continue
            f1s = []
            for nf in files:
                cid = nf.stem.replace("_notesA1", "")
                wav = audio_dir / f"{cid}.wav"
                if not wav.exists(): continue
                gt_iv, gt_p = load_notes(nf)
                notes = predict(wav, vt_on, vt_off, psw=19, mns=0.052, oms=0.026)
                _, _, f = score(notes, gt_iv, gt_p)
                f1s.append(f)
            mf = float(np.mean(f1s))
            rows.append({"vt_on": vt_on, "vt_off": vt_off, "f1": mf})
            print(f"  vt_on={vt_on:.2f} vt_off={vt_off:.2f}  F1={mf:.3f}")
            wandb.log({"vt_on": vt_on, "vt_off": vt_off, "f1": mf})
    rows.sort(key=lambda r: -r["f1"])
    print(f"\nTop 5:")
    for r in rows[:5]:
        print(f"  F1={r['f1']:.3f}  vt_on={r['vt_on']} vt_off={r['vt_off']}")
    out = Path("reports/_exp_B47_hysteresis.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
