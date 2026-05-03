"""B58: disambiguate upstream loss on Romantic ASAP.
Three configs per piece:
  A. GT notes + GT beats (B53 oracle, both perfect)
  B. GT notes + beat_this beats (only beats are noisy)
  C. ByteDance notes + GT beats (only notes are noisy)
This tells us how much of the upstream loss comes from notes vs beats."""
from __future__ import annotations
import json, subprocess, pickle
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
CACHE_DIR = Path("/workspace/.cache/asap_bytedance")
ALLOWED_BEATS = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24


def load_cached(piece_dir: Path):
    key = piece_dir.relative_to(ASAP).as_posix().replace("/", "__")
    cache_path = CACHE_DIR / f"{key}.pkl"
    with open(cache_path, "rb") as f:
        d = pickle.load(f)
    notes = []
    for x in d["notes"]:
        hz = x.get("hz")
        midi = x.get("midi")
        if hz is None and midi is not None:
            hz = 440.0 * 2 ** ((midi - 69) / 12)
        if hz is None or midi is None:
            continue
        notes.append(NoteEvent(onset_s=x["on"], offset_s=x["off"], pitch_midi=midi, pitch_hz=hz, velocity=x.get("vel", 80), confidence=x.get("conf", 1.0)))
    return notes, d["beats"]


def load_midi_events(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    events = []; iv = []; pi = []
    for inst in pm.instruments:
        for n in inst.notes:
            hz = 440.0 * 2 ** ((n.pitch - 69) / 12)
            events.append(NoteEvent(onset_s=n.start, offset_s=n.end,
                                    pitch_midi=n.pitch, pitch_hz=hz,
                                    velocity=n.velocity, confidence=1.0))
            iv.append([n.start, n.end]); pi.append(hz)
    events.sort(key=lambda e: e.onset_s)
    return events, np.array(iv), np.array(pi), pm.get_beats()


def snap(d): return float(ALLOWED_BEATS[np.argmin(np.abs(ALLOWED_BEATS - d))])


def evaluate(notes, beats, gt_iv, gt_p, avg_beat):
    pj = adaptive_pitch_jump(notes)
    cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on, q_off = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=TPB,
                                           allowed_durations_tatums=default_allowed_durations(TPB))
    pred_durs = (q_off - q_on) / TPB
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    onsets = np.array([n.onset_s for n in notes]); offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([n.pitch_hz for n in notes])
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, est_p, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if not matched: return 0.0, pj
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    return float(np.mean(pd_s == gd_s)), pj


PIECES = ["Bach/Fugue/bwv_846", "Beethoven/Piano_Sonatas/21-1",
          "Schumann/Toccata", "Chopin/Berceuse_op_57"]


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B58_disambig_upstream",
                     config={"git_sha": git_sha()}, tags=["B58", "asap", "diag"], dir="logs/wandb")
    rows = []
    print("\n  piece                              A:gt+gt  B:gt+bt  C:bd+gt  D:bd+bt")
    print("  " + "-" * 80)
    for piece_rel in PIECES:
        d = ASAP / piece_rel
        cached = CACHE_DIR / (piece_rel.replace("/", "__") + ".pkl")
        if not (d / "midi_score.mid").exists() or not cached.exists():
            print(f"  {piece_rel}: missing"); continue
        gt_events, gt_iv, gt_p, gt_beats = load_midi_events(d / "midi_score.mid")
        gt_avg = float(np.diff(gt_beats).mean())
        bd_notes, bd_beats = load_cached(d)
        bd_avg = float(np.diff(bd_beats).mean())
        # A: GT notes + GT beats
        sn_A, _ = evaluate(gt_events, gt_beats, gt_iv, gt_p, gt_avg)
        # B: GT notes + beat_this beats
        sn_B, _ = evaluate(gt_events, bd_beats, gt_iv, gt_p, bd_avg)
        # C: ByteDance notes + GT beats
        sn_C, _ = evaluate(bd_notes, gt_beats, gt_iv, gt_p, gt_avg)
        # D: ByteDance notes + beat_this beats (current production)
        sn_D, _ = evaluate(bd_notes, bd_beats, gt_iv, gt_p, bd_avg)
        rows.append({"piece": piece_rel, "A_gt_gt": sn_A, "B_gt_bt": sn_B, "C_bd_gt": sn_C, "D_bd_bt": sn_D})
        print(f"  {piece_rel:35s}  {sn_A:.3f}    {sn_B:.3f}    {sn_C:.3f}    {sn_D:.3f}")
        wandb.log({piece_rel: {"A": sn_A, "B": sn_B, "C": sn_C, "D": sn_D}})
    if not rows:
        print("nothing"); run.finish(); return
    means = {k: float(np.mean([r[k] for r in rows])) for k in ("A_gt_gt", "B_gt_bt", "C_bd_gt", "D_bd_bt")}
    print("\nMeans:")
    for k, v in means.items(): print(f"  {k:12s} = {v:.3f}")
    print(f"\nLoss decomposition (mean):")
    print(f"  Total upstream loss     = {means['A_gt_gt'] - means['D_bd_bt']:.3f}")
    print(f"  ByteDance-only loss (A->C) = {means['A_gt_gt'] - means['C_bd_gt']:.3f}")
    print(f"  beat_this-only loss (A->B) = {means['A_gt_gt'] - means['B_gt_bt']:.3f}")
    wandb.summary.update(means)
    out = Path("reports/_exp_B58_disambig.json")
    out.write_text(json.dumps({"per_piece": rows, "means": means}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
