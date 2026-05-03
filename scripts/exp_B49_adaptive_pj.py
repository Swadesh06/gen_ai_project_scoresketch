"""B49: per-piece adaptive pitch_jump (auto-select based on note density + pitch spread).
Uses cached ByteDance from B48 for speed."""
from __future__ import annotations
import json, subprocess, pickle
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24
CACHE_DIR = Path("/workspace/.cache/asap_bytedance")


def load_cached(piece_dir: Path):
    key = piece_dir.relative_to(ASAP).as_posix().replace("/", "__")
    cache_path = CACHE_DIR / f"{key}.pkl"
    with open(cache_path, "rb") as f:
        d = pickle.load(f)
    notes = [NoteEvent(onset_s=x["on"], offset_s=x["off"], pitch_midi=x["midi"], pitch_hz=x["hz"], velocity=x["vel"], confidence=x["conf"]) for x in d["notes"]]
    return notes, d["beats"]


def load_midi_notes(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv = []; pi = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv, dtype=np.float64), np.array(pi, dtype=np.float64)


def snap(d): return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


PIECES = ["Bach/Fugue/bwv_846", "Bach/Fugue/bwv_848", "Bach/Fugue/bwv_854",
          "Bach/Fugue/bwv_856", "Bach/Fugue/bwv_857",
          "Beethoven/Piano_Sonatas/21-1", "Schumann/Toccata",
          "Chopin/Berceuse_op_57", "Liszt/Sonata"]


def eval_piece(piece_dir: Path, pj: float):
    notes, beats = load_cached(piece_dir)
    avg_beat = float(np.diff(beats).mean()) if len(beats) >= 2 else 0.5
    cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on, q_off = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=TPB,
                                           allowed_durations_tatums=default_allowed_durations(TPB))
    pred_durs = (q_off - q_on) / TPB
    mid = piece_dir / "midi_score.mid"
    gt_iv, gt_p = load_midi_notes(mid)
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    onsets = np.array([n.onset_s for n in notes]); offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([440.0 * 2 ** ((n.midi() - 69) / 12) for n in notes])
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, est_p,
                                                  onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if not matched: return 0.0, 0.0
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    raw = float(np.mean(np.abs(pd - gd) < 0.05))
    snap_pct = float(np.mean(pd_s == gd_s))
    return raw, snap_pct


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B49_adaptive_pj",
                     config={"git_sha": git_sha()}, tags=["B49", "asap", "adaptive"], dir="logs/wandb")
    rows = []
    for pj_strategy in ("fixed_3", "fixed_7", "fixed_12", "adaptive"):
        per = []
        for piece_rel in PIECES:
            d = ASAP / piece_rel
            cache_path = CACHE_DIR / (piece_rel.replace("/", "__") + ".pkl")
            if not cache_path.exists():
                # Need to compute first — skip for now if not cached
                continue
            try:
                notes, beats = load_cached(d)
                if pj_strategy.startswith("fixed_"):
                    pj = float(pj_strategy.split("_")[1])
                else:  # adaptive
                    pj = adaptive_pitch_jump(notes)
                raw, snap_pct = eval_piece(d, pj)
            except Exception as e:
                print(f"  {piece_rel}: failed -- {e}"); continue
            per.append({"piece": piece_rel, "raw": raw, "snap": snap_pct, "pj": pj})
        if not per: continue
        mean_snap = float(np.mean([r["snap"] for r in per]))
        rows.append({"strategy": pj_strategy, "mean_snap": mean_snap, "per": per})
        print(f"\n{pj_strategy:18s}  mean_snap={mean_snap:.3f}  (n={len(per)})")
        for r in per:
            print(f"   {r['piece']:35s}  pj={r['pj']:5.1f}  snap={r['snap']:.3f}")
        wandb.log({f"{pj_strategy}/mean_snap": mean_snap})
    rows.sort(key=lambda r: -r["mean_snap"])
    print(f"\nTop strategies:")
    for r in rows:
        print(f"  mean_snap={r['mean_snap']:.3f}  {r['strategy']}")
    out = Path("reports/_exp_B49_adaptive_pj.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
