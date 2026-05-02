"""Exp B11: voicing + HMM ensemble. Try intersection (precision-bias) and
union (recall-bias). Compare to either method alone on Vocadito A1."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.hmm_segment import HMMConfig, segment_pitch_to_notes_hmm
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC = Path("~/datasets/vocadito").expanduser()
BEST_HMM = HMMConfig(
    midi_lo=36, midi_hi=96,
    p_sustain=0.878, p_end=0.036, p_start=0.094,
    sigma_voicing=0.22, sigma_midi=0.97, interval_decay=0.73,
)


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def to_arrays(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    hz = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    return iv, hz


def score(iv_pred: np.ndarray, hz_pred: np.ndarray, iv_gt: np.ndarray, hz_gt: np.ndarray) -> tuple[float, float, float]:
    if len(iv_pred) == 0:
        return 0.0, 0.0, 0.0
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv_gt, hz_gt, iv_pred, hz_pred, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return float(p), float(r), float(f)


def intersect_notes(a: list[NoteEvent], b: list[NoteEvent], onset_tol: float = 0.05) -> list[NoteEvent]:
    """Keep notes from a that have a matching note in b (onset and pitch within tol)."""
    if not a or not b:
        return []
    out: list[NoteEvent] = []
    b_on = np.array([n.onset_s for n in b])
    b_pi = np.array([n.midi() for n in b])
    for na in a:
        match = (np.abs(b_on - na.onset_s) <= onset_tol) & (np.abs(b_pi - na.midi()) <= 1)
        if match.any():
            out.append(na)
    return out


def union_notes(a: list[NoteEvent], b: list[NoteEvent], onset_tol: float = 0.05) -> list[NoteEvent]:
    """Combine; for overlapping pairs, keep the higher-confidence note."""
    out: list[NoteEvent] = []
    used_b = set()
    for i, na in enumerate(a):
        best_j = None
        for j, nb in enumerate(b):
            if j in used_b:
                continue
            if abs(nb.onset_s - na.onset_s) <= onset_tol and abs(nb.midi() - na.midi()) <= 1:
                best_j = j; break
        if best_j is not None:
            used_b.add(best_j)
            chosen = na if na.confidence >= b[best_j].confidence else b[best_j]
            out.append(chosen)
        else:
            out.append(na)
    for j, nb in enumerate(b):
        if j not in used_b:
            out.append(nb)
    out.sort(key=lambda n: n.onset_s)
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    cfg = {"exp": "B11_voicing_hmm_ensemble", "git_sha": git_sha()}
    run = wandb.init(
        project="humscribe-v3.2", name="exp_B11_ensemble",
        config=cfg, tags=["B11", "vocadito", "ensemble"], dir="logs/wandb",
    )
    notes_dir = VOC / "Annotations" / "Notes"
    audio_dir = VOC / "Audio"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    mc = ModeConfig.for_mode("soft")
    rows: dict[str, list[float]] = {"voicing": [], "hmm": [], "intersect": [], "union": []}
    rows_p: dict[str, list[float]] = {k: [] for k in rows}
    rows_r: dict[str, list[float]] = {k: [] for k in rows}
    for nf in files:
        cid = nf.stem.replace("_notesA1", "")
        wav = audio_dir / f"{cid}.wav"
        if not wav.exists():
            continue
        audio, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_pesto(audio, sr)
        v_notes = segment_pitch_to_notes(t, hz, vc, mc)
        v_notes = [n for n in v_notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]
        h_notes = segment_pitch_to_notes_hmm(t, hz, vc, mc, BEST_HMM)
        h_notes = [n for n in h_notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]
        i_notes = intersect_notes(v_notes, h_notes)
        u_notes = union_notes(v_notes, h_notes)

        gt_iv, gt_p = load_notes(nf)
        for name, ns in (("voicing", v_notes), ("hmm", h_notes), ("intersect", i_notes), ("union", u_notes)):
            iv, hzp = to_arrays(ns)
            p, r, f = score(iv, hzp, gt_iv, gt_p)
            rows[name].append(f); rows_p[name].append(p); rows_r[name].append(r)

    print(f"\n{'method':>10s}  {'F1':>5s}  {'P':>5s}  {'R':>5s}")
    summary = {}
    for name in rows:
        mf = float(np.mean(rows[name])); mp = float(np.mean(rows_p[name])); mr = float(np.mean(rows_r[name]))
        print(f"  {name:>10s}  {mf:.3f}  {mp:.3f}  {mr:.3f}")
        summary[f"{name}/F1"] = mf
        summary[f"{name}/P"] = mp
        summary[f"{name}/R"] = mr
    wandb.summary.update(summary)
    out = Path("reports/_exp_B11_ensemble.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nrun: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
