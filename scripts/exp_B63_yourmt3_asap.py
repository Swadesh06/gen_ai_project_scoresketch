"""B63 — YourMT3+ vs ByteDance on the 9-piece ASAP test set (B+2 item 2).

Runs both transcribers through the rest of the pipeline (DP+VT, beat_this beats)
and compares Stage-5 snap. Targets from `task_description_v2.md` §Work item 2:
- Beethoven snap ≥ 0.92 (oracle 0.982)
- 5-mixed mean ≥ 0.74 (oracle 0.905)
- Schumann snap ≥ 0.93 (oracle 0.975)

Decision rule: if Beethoven ≥ 0.85 AND mixed-mean ≥ 0.70, promote YourMT3+ as
default for Romantic-detected pieces (auto_piano routing).

Caches transcriber outputs per piece per backend at
`/workspace/.cache/asap_<backend>/<key>.pkl` so subsequent runs are fast.
"""
from __future__ import annotations
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import (
    VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations,
)


ASAP = Path("~/datasets/asap").expanduser()
RENDERS = Path("/workspace/.cache/asap_renders")
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
ALLOWED_BEATS = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24

# 5-Bach Fugue baseline + 4 Romantic
PIECES = [
    "Bach/Fugue/bwv_846", "Bach/Fugue/bwv_848", "Bach/Fugue/bwv_854",
    "Bach/Fugue/bwv_856", "Bach/Fugue/bwv_857",
    "Beethoven/Piano_Sonatas/21-1", "Schumann/Toccata",
    "Chopin/Berceuse_op_57", "Liszt/Sonata",
]


def piece_key(p: str) -> str:
    return p.replace("/", "__")


def render_audio(piece_rel: str) -> Path:
    """Use the ASAP repo's bundled midi_score.wav if present; else fall back to
    the cache; else render via fluidsynth."""
    src = ASAP / piece_rel / "midi_score.wav"
    dst = RENDERS / f"{piece_key(piece_rel)}.wav"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    if src.exists() and src.stat().st_size > 0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return dst
    mid = ASAP / piece_rel / "midi_score.mid"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(dst), "-T", "wav",
                    SF2, str(mid)], check=True, capture_output=True)
    return dst


def cached_transcribe(piece_rel: str, backend: str) -> tuple[list[NoteEvent], np.ndarray, float]:
    cache_dir = Path(f"/workspace/.cache/asap_{backend}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{piece_key(piece_rel)}.pkl"
    if cp.exists():
        with cp.open("rb") as f:
            d = pickle.load(f)
        notes = []
        for x in d["notes"]:
            hz = x.get("hz")
            mid = x.get("midi")
            if hz is None and mid is not None:
                hz = 440.0 * 2 ** ((mid - 69) / 12)
            if hz is None or mid is None:
                continue
            notes.append(NoteEvent(onset_s=x["on"], offset_s=x["off"],
                                   pitch_midi=mid, pitch_hz=hz,
                                   velocity=x.get("vel", 80),
                                   confidence=x.get("conf", 1.0)))
        return notes, np.asarray(d["beats"]), float(d.get("bpm", 120.0))
    wav = render_audio(piece_rel)
    if backend == "bytedance":
        notes = transcribe_piano(str(wav))
    elif backend == "yourmt3plus":
        from humscribe.instrument.yourmt3plus import transcribe_yourmt3plus
        notes = transcribe_yourmt3plus(str(wav))
    else:
        raise ValueError(backend)
    beats, _, bpm = track_beats_beat_this(str(wav))
    notes_data = []
    for n in notes:
        hz = getattr(n, "pitch_hz", None)
        mid = getattr(n, "pitch_midi", None)
        if hz is None and mid is not None:
            hz = 440.0 * 2 ** ((mid - 69) / 12)
        notes_data.append({
            "on": float(n.onset_s), "off": float(n.offset_s),
            "midi": int(mid) if mid is not None else None,
            "hz": float(hz) if hz is not None else None,
            "vel": int(getattr(n, "velocity", 80)),
            "conf": float(getattr(n, "confidence", 1.0)),
        })
    with cp.open("wb") as f:
        pickle.dump({"notes": notes_data, "beats": np.asarray(beats), "bpm": float(bpm)}, f)
    # rewrite without None pitch_hz to match downstream expectations
    cleaned = [n for n in notes if (n.pitch_hz is not None or n.pitch_midi is not None)]
    return cleaned, np.asarray(beats), float(bpm)


def load_midi_notes(mid: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv: list[list[float]] = []
    pi: list[float] = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv), np.array(pi)


def snap(d: float) -> float:
    return float(ALLOWED_BEATS[np.argmin(np.abs(ALLOWED_BEATS - d))])


def evaluate(notes: list[NoteEvent], beats: np.ndarray,
             gt_iv: np.ndarray, gt_p: np.ndarray) -> dict:
    if not notes or len(beats) < 2:
        return {"snap": 0.0, "raw": 0.0, "n_pred": len(notes), "n_matched": 0}
    avg_beat = float(np.diff(beats).mean())
    pj = adaptive_pitch_jump(notes)
    cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on, q_off = viterbi_quantize_rhythm(
        on_v, off_v, beats, tatums_per_beat=TPB,
        allowed_durations_tatums=default_allowed_durations(TPB),
    )
    pred_durs = (q_off - q_on) / TPB
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    def _hz(n: NoteEvent) -> float | None:
        if n.pitch_hz is not None and n.pitch_hz > 0:
            return float(n.pitch_hz)
        if n.pitch_midi is not None:
            return 440.0 * 2 ** ((int(n.pitch_midi) - 69) / 12)
        return None
    valid_idx = [i for i, n in enumerate(notes) if _hz(n) is not None]
    if not valid_idx:
        return {"snap": 0.0, "raw": 0.0, "n_pred": len(notes), "n_matched": 0}
    onsets = np.array([notes[i].onset_s for i in valid_idx])
    offsets = np.array([notes[i].offset_s for i in valid_idx])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([_hz(notes[i]) for i in valid_idx])
    pred_durs = pred_durs[valid_idx]
    matched = mir_eval.transcription.match_notes(
        gt_iv, gt_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    if not matched:
        return {"snap": 0.0, "raw": 0.0, "n_pred": len(notes), "n_matched": 0}
    gi = [m[0] for m in matched]
    pi = [m[1] for m in matched]
    pd = pred_durs[pi]
    gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    raw = float(np.mean(np.abs(pd - gd) < 0.05))
    snap_pct = float(np.mean(pd_s == gd_s))
    return {"snap": snap_pct, "raw": raw, "n_pred": len(notes), "n_matched": len(matched)}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(only: list[str]) -> None:
    cfg_w = {"git_sha": git_sha(), "pieces": only or PIECES}
    run = wandb.init(project="humscribe-v3.2", name="exp_B63_yourmt3_asap",
                     config=cfg_w, tags=["B63", "asap", "yourmt3plus", "item2"],
                     dir="logs/wandb")
    rows = []
    targets = only if only else PIECES
    print(f"\n  {'piece':35s}  bd_snap  ymt3_snap  Δ      bd_n  ymt3_n")
    print("  " + "-" * 80)
    for piece_rel in targets:
        if not (ASAP / piece_rel).exists():
            print(f"  {piece_rel}: not found"); continue
        try:
            bd_notes, bd_beats, _bpm_b = cached_transcribe(piece_rel, "bytedance")
            ymt3_notes, ymt3_beats, _bpm_y = cached_transcribe(piece_rel, "yourmt3plus")
            gt_iv, gt_p = load_midi_notes(ASAP / piece_rel / "midi_score.mid")
            r_bd = evaluate(bd_notes, bd_beats, gt_iv, gt_p)
            r_y = evaluate(ymt3_notes, ymt3_beats, gt_iv, gt_p)
        except Exception as e:
            print(f"  {piece_rel}: FAILED -- {e}"); continue
        delta = r_y["snap"] - r_bd["snap"]
        rows.append({"piece": piece_rel, "bd_snap": r_bd["snap"], "ymt3_snap": r_y["snap"],
                     "delta": delta, "bd_n": r_bd["n_pred"], "ymt3_n": r_y["n_pred"]})
        print(f"  {piece_rel:35s}  {r_bd['snap']:.3f}    {r_y['snap']:.3f}    {delta:+.3f}  "
              f"{r_bd['n_pred']:5d}  {r_y['n_pred']:5d}")
        wandb.log({piece_rel: {"bd_snap": r_bd["snap"], "ymt3_snap": r_y["snap"]}})
    if rows:
        bach_rows = [r for r in rows if r["piece"].startswith("Bach/")]
        rom_rows = [r for r in rows if not r["piece"].startswith("Bach/") and "Liszt" not in r["piece"]]
        bd_mean = float(np.mean([r["bd_snap"] for r in rows]))
        y_mean = float(np.mean([r["ymt3_snap"] for r in rows]))
        bd_bach_mean = float(np.mean([r["bd_snap"] for r in bach_rows])) if bach_rows else 0
        y_bach_mean = float(np.mean([r["ymt3_snap"] for r in bach_rows])) if bach_rows else 0
        bd_rom_mean = float(np.mean([r["bd_snap"] for r in rom_rows])) if rom_rows else 0
        y_rom_mean = float(np.mean([r["ymt3_snap"] for r in rom_rows])) if rom_rows else 0
        summary = {
            "bd_overall_mean": bd_mean, "ymt3_overall_mean": y_mean,
            "bd_bach_mean": bd_bach_mean, "ymt3_bach_mean": y_bach_mean,
            "bd_romantic_mean": bd_rom_mean, "ymt3_romantic_mean": y_rom_mean,
            "n_pieces": len(rows),
        }
        print(f"\n  Summary across {len(rows)} pieces:")
        for k, v in summary.items():
            print(f"    {k:24s} = {v}")
        wandb.summary.update(summary)
        Path("reports/_exp_B63_yourmt3_asap.json").write_text(
            json.dumps({"rows": rows, "summary": summary, "config": cfg_w}, indent=2))
    print(f"\n  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[],
                    help="subset of pieces (e.g. Bach/Fugue/bwv_846)")
    main(only=ap.parse_args().only)
