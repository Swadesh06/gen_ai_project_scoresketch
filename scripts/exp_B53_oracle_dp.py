"""B53: oracle-input DP test on Romantic ASAP.
Feed GT MIDI notes + GT beats directly into DP. If snap is still bad,
the bottleneck is DP. If snap is high, the bottleneck is upstream
(ByteDance/beat_this) on Romantic input."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24


def load_score_beats(ann: Path):
    return np.array(sorted(float(line.split()[0]) for line in ann.read_text().splitlines() if line.split()))


def load_midi_notes_as_events(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            hz = 440.0 * 2 ** ((n.pitch - 69) / 12)
            notes.append(NoteEvent(onset_s=n.start, offset_s=n.end,
                                   pitch_midi=n.pitch, pitch_hz=hz, velocity=n.velocity, confidence=1.0))
    notes.sort(key=lambda e: e.onset_s)
    return notes


def snap(d): return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


PIECES = ["Bach/Fugue/bwv_846", "Beethoven/Piano_Sonatas/21-1",
          "Schumann/Toccata", "Chopin/Berceuse_op_57", "Liszt/Sonata"]


def find_perf(piece_dir: Path) -> Path | None:
    """Find a *.mid (recorded performance) for beat extraction (use first)."""
    for p in piece_dir.iterdir():
        if p.suffix == ".mid" and p.stem != "midi_score":
            return p
    return None


def find_annot(piece_dir: Path) -> Path | None:
    """Find a *.txt annotation file (beat positions for the performance)."""
    for p in piece_dir.iterdir():
        if p.suffix == ".txt" and "annot" in p.name.lower():
            return p
    return None


def eval_oracle(piece_dir: Path):
    mid = piece_dir / "midi_score.mid"
    if not mid.exists(): return None
    # Use score MIDI notes as both GT and "predicted" — pure DP test
    notes = load_midi_notes_as_events(mid)
    if len(notes) < 5: return None
    # Generate beats from GT MIDI: each beat = quarter note from MIDI tempo
    pm = pretty_midi.PrettyMIDI(str(mid))
    end = pm.get_end_time()
    beats = pm.get_beats()  # built-in beat extraction
    if len(beats) < 4: return None
    avg_beat = float(np.diff(beats).mean())
    pj = adaptive_pitch_jump(notes)
    cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on, q_off = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=TPB,
                                           allowed_durations_tatums=default_allowed_durations(TPB))
    pred_durs = (q_off - q_on) / TPB
    # GT durations (in beats)
    gt_iv = np.array([[n.onset_s, n.offset_s] for n in notes])
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    # Use mir_eval matched-pair to align (here predictions ARE GT, so should perfectly match)
    gt_p = np.array([n.pitch_hz for n in notes])
    onsets = np.array([n.onset_s for n in notes])
    offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, gt_p,
                                                  onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if not matched: return 0.0, 0.0, pj
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    raw = float(np.mean(np.abs(pd - gd) < 0.05))
    snap_pct = float(np.mean(pd_s == gd_s))
    return raw, snap_pct, pj


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B53_oracle_dp",
                     config={"git_sha": git_sha()}, tags=["B53", "asap", "oracle"], dir="logs/wandb")
    rows = []
    print("Oracle test: GT MIDI notes + GT MIDI beats fed to DP")
    print("-" * 70)
    for piece_rel in PIECES:
        d = ASAP / piece_rel
        if not d.exists(): print(f"  {piece_rel}: not found"); continue
        try:
            r = eval_oracle(d)
            if r is None: print(f"  {piece_rel}: no MIDI score"); continue
            raw, snap_pct, pj = r
            rows.append({"piece": piece_rel, "raw": raw, "snap": snap_pct, "pj": pj})
            print(f"  {piece_rel:35s}  pj={pj:5.1f}  raw={raw:.3f}  snap={snap_pct:.3f}")
            wandb.log({piece_rel: snap_pct})
        except Exception as e:
            print(f"  {piece_rel}: failed -- {e}")
    if rows:
        mean_snap = float(np.mean([r["snap"] for r in rows]))
        print(f"\nOracle mean snap = {mean_snap:.3f}  (n={len(rows)})")
        wandb.summary.update({"oracle_mean_snap": mean_snap, "n": len(rows)})
        out = Path("reports/_exp_B53_oracle_dp.json")
        out.write_text(json.dumps({"rows": rows, "oracle_mean_snap": mean_snap}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
