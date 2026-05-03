"""B62 — voicing exit-side hysteresis sweep on Vocadito (B+2 work item 4).

Caches PESTO + CREPE pitch traces per clip once, then sweeps vt_exit ∈
{0.25, 0.35, 0.45, 0.55, 0.65} with vt_enter=0.75 fixed. Reports
no_offset / offset50 / offset20 F1 against A1 and A2 separately so we know
both that the offset gap closes (target) and that no-offset doesn't regress
(decision-rule guard).

Pass criteria (per task_description_v2.md item 4):
  Vocadito A1 no_offset F1   ≥ 0.66 (no regression vs 0.665)
  Vocadito A1 offset50 F1    ≥ 0.60 (vs 0.573)
  Vocadito A1 offset20 F1    ≥ 0.50 (vs 0.439)
"""
from __future__ import annotations
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.notes import NoteEvent, hz_to_midi, midi_to_hz
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto


VOC = Path("~/datasets/vocadito").expanduser()
CACHE = Path("/workspace/.cache/vocadito_pitch")
CACHE.mkdir(parents=True, exist_ok=True)


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def cached_traces(clip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (times, hz_pesto, voicing_hybrid)."""
    cp = CACHE / f"{clip_id}.pkl"
    if cp.exists():
        with cp.open("rb") as f:
            d = pickle.load(f)
        return d["t"], d["hz"], d["voicing"]
    wav = VOC / "Audio" / f"{clip_id}.wav"
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, _pv = track_pitch_pesto(audio, sr)
    ct, _ch, cv = track_pitch_crepe(audio, sr)
    voicing = np.interp(pt, ct, cv)
    with cp.open("wb") as f:
        pickle.dump({"t": pt, "hz": ph, "voicing": voicing}, f)
    return pt, ph, voicing


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


def segment_hysteresis(times: np.ndarray, hz: np.ndarray, voicing: np.ndarray,
                       vt_enter: float, vt_exit: float, psw: int,
                       mns: float, oms: float) -> list[NoteEvent]:
    if len(times) == 0:
        return []
    midi = np.where(hz > 0, np.array([hz_to_midi(float(h)) for h in hz]), 0.0)
    smooth = median_filter(midi, psw)
    n = len(times)
    states = np.zeros(n, dtype=bool)
    in_note = False
    for i in range(n):
        if not in_note and voicing[i] >= vt_enter:
            in_note = True
        elif in_note and voicing[i] < vt_exit:
            in_note = False
        states[i] = in_note
    segs = []
    i = 0
    while i < n:
        if not states[i]:
            i += 1; continue
        j = i
        while j + 1 < n and states[j + 1]:
            j += 1
        segs.append((i, j)); i = j + 1
    if not segs:
        return []
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if times[s] - times[pe] < oms:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    notes: list[NoteEvent] = []
    for s, e in merged:
        sub_t = times[s:e + 1]; sub_m = smooth[s:e + 1]; sub_v = voicing[s:e + 1]
        start = 0
        cur_med = float(np.median(sub_m[:max(int(len(sub_m) * 0.2), 1)]))
        for k in range(1, len(sub_t)):
            if abs(float(sub_m[k]) - cur_med) > 0.5:
                _emit_note(sub_t, sub_m, sub_v, start, k - 1, mns, notes)
                start = k
                cur_med = float(sub_m[k])
        _emit_note(sub_t, sub_m, sub_v, start, len(sub_t) - 1, mns, notes)
    return notes


def _emit_note(t, m, v, s: int, e: int, mns: float, out: list[NoteEvent]) -> None:
    if e <= s:
        return
    midi_med = float(np.median(m[s:e + 1]))
    midi_int = int(round(midi_med)) if midi_med > 0 else 0
    if midi_int <= 0:
        return
    on_t = float(t[s])
    off_t = float(t[e]) + (float(t[1] - t[0]) if len(t) > 1 else 0.01)
    if (off_t - on_t) < mns:
        return
    out.append(NoteEvent(onset_s=on_t, offset_s=off_t,
                         pitch_hz=midi_to_hz(midi_med), pitch_midi=midi_int,
                         confidence=float(np.mean(v[s:e + 1]))))


def score(notes: list[NoteEvent], ref_iv: np.ndarray, ref_p: np.ndarray,
          offset_ratio: float | None) -> float:
    if not notes:
        return 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    ep = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, eiv, ep,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=offset_ratio, offset_min_tolerance=0.05,
    )
    return float(f)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vt-enter", type=float, default=0.75)
    ap.add_argument("--exits", default="0.25,0.35,0.45,0.55,0.65")
    ap.add_argument("--psw", type=int, default=19)
    ap.add_argument("--mns", type=float, default=0.052)
    ap.add_argument("--oms", type=float, default=0.026)
    args = ap.parse_args()

    notes_dir = VOC / "Annotations" / "Notes"
    a1_files = sorted(notes_dir.glob("*_notesA1.csv"))
    print(f"Vocadito clips with A1 ann: {len(a1_files)}")
    print("Caching pitch+voicing traces (PESTO + CREPE)…")
    for f in a1_files:
        cid = f.stem.replace("_notesA1", "")
        cached_traces(cid)
    print("  cache populated.")

    exits = [float(x) for x in args.exits.split(",")]
    cfg_w = {"git_sha": git_sha(), "vt_enter": args.vt_enter, "exits": exits,
             "psw": args.psw, "mns": args.mns, "oms": args.oms}
    run = wandb.init(project="humscribe-v3.2", name="exp_B62_voc_exit_hysteresis",
                     config=cfg_w, tags=["B62", "vocadito", "hysteresis", "item4"],
                     dir="logs/wandb")
    rows = []
    print(f"\n  vt_exit  | A1 no/o50/o20  | A2 no/o50/o20")
    print("  ---------+----------------+----------------")
    for vte in exits:
        a1_no, a1_o50, a1_o20 = [], [], []
        a2_no, a2_o50, a2_o20 = [], [], []
        for f in a1_files:
            cid = f.stem.replace("_notesA1", "")
            t, hz, voicing = cached_traces(cid)
            notes = segment_hysteresis(t, hz, voicing, args.vt_enter, vte,
                                        args.psw, args.mns, args.oms)
            a1_iv, a1_p = load_notes(f)
            a1_no.append(score(notes, a1_iv, a1_p, None))
            a1_o50.append(score(notes, a1_iv, a1_p, 0.5))
            a1_o20.append(score(notes, a1_iv, a1_p, 0.2))
            a2 = notes_dir / f"{cid}_notesA2.csv"
            if a2.exists():
                a2_iv, a2_p = load_notes(a2)
                a2_no.append(score(notes, a2_iv, a2_p, None))
                a2_o50.append(score(notes, a2_iv, a2_p, 0.5))
                a2_o20.append(score(notes, a2_iv, a2_p, 0.2))
        m = lambda xs: float(np.mean(xs)) if xs else 0.0
        row = {
            "vt_exit": vte,
            "A1_no": m(a1_no), "A1_o50": m(a1_o50), "A1_o20": m(a1_o20),
            "A2_no": m(a2_no), "A2_o50": m(a2_o50), "A2_o20": m(a2_o20),
        }
        rows.append(row)
        print(f"  {vte:7.2f}  | {row['A1_no']:.3f} {row['A1_o50']:.3f} {row['A1_o20']:.3f}"
              f"  | {row['A2_no']:.3f} {row['A2_o50']:.3f} {row['A2_o20']:.3f}")
        wandb.log(row)
    out = Path("reports/_exp_B62_voc_exit_hysteresis.json")
    out.write_text(json.dumps({"rows": rows, "config": cfg_w}, indent=2))
    print(f"\n  wrote {out}")
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
