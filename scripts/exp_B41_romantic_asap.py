"""B41: ASAP non-Bach with adapted VT params (wider pitch_jump, longer time_gap).
Romantic music has wider melodic intervals + thicker chordal textures."""
from __future__ import annotations
import json, subprocess, shutil
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24


def render(midi: Path, wav: Path):
    if wav.exists() and wav.stat().st_size > 0: return
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)], check=True, capture_output=True)


def load_score_beats(ann: Path):
    return np.array(sorted(float(line.split()[0]) for line in ann.read_text().splitlines() if line.split()))


def load_midi_notes(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv = []; pi = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv, dtype=np.float64), np.array(pi, dtype=np.float64)


def snap(d): return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


def eval_piece(piece_dir: Path, pj: float, tg: float, no_vt: bool = False):
    mid = piece_dir / "midi_score.mid"
    wav = piece_dir / "midi_score.wav"
    ann = piece_dir / "midi_score_annotations.txt"
    render(mid, wav)
    beats = load_score_beats(ann)
    avg_beat = float(np.diff(beats).mean()) if len(beats) >= 2 else 0.5
    notes = transcribe_piano(str(wav))
    onsets = np.array([n.onset_s for n in notes]); offsets = np.array([n.offset_s for n in notes])
    if no_vt:
        q_on, q_off = viterbi_quantize_rhythm(onsets, offsets, beats, tatums_per_beat=TPB,
                                               allowed_durations_tatums=default_allowed_durations(TPB))
    else:
        cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=tg)
        voices = assign_voices(notes, cfg)
        on_v, off_v = per_voice_durations(notes, voices)
        q_on, q_off = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=TPB,
                                               allowed_durations_tatums=default_allowed_durations(TPB))
    pred_durs = (q_off - q_on) / TPB
    gt_iv, gt_p = load_midi_notes(mid)
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
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


PIECES = ["Beethoven/Piano_Sonatas/21-1", "Schumann/Toccata",
          "Chopin/Berceuse_op_57", "Liszt/Sonata"]


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B41_romantic",
                      config={"git_sha": git_sha()}, tags=["B41", "asap", "romantic"], dir="logs/wandb")
    rows = []
    configs = [("greedy_default", 3.0, 0.5, False),
               ("wider_pj7", 7.0, 0.5, False),
               ("wider_pj12", 12.0, 1.0, False),
               ("wide_pj9_tg2", 9.0, 2.0, False),
               ("no_vt", 0, 0, True)]
    for name, pj, tg, no_vt in configs:
        per = []
        for piece_rel in PIECES:
            d = ASAP / piece_rel
            if not (d / "midi_score.mid").exists():
                continue
            try:
                raw, snap_pct = eval_piece(d, pj, tg, no_vt)
            except Exception as e:
                print(f"  {piece_rel}: failed -- {e}"); continue
            per.append({"piece": piece_rel, "raw": raw, "snap": snap_pct})
        if not per: continue
        mean_snap = float(np.mean([r["snap"] for r in per]))
        mean_raw = float(np.mean([r["raw"] for r in per]))
        rows.append({"config": name, "pj": pj, "tg": tg, "no_vt": no_vt,
                     "mean_snap": mean_snap, "mean_raw": mean_raw, "per": per})
        print(f"{name:18s}  mean_snap={mean_snap:.3f}  mean_raw={mean_raw:.3f}")
        for r in per:
            print(f"   {r['piece']:35s}  snap={r['snap']:.3f}  raw={r['raw']:.3f}")
        wandb.log({f"{name}/mean_snap": mean_snap, f"{name}/mean_raw": mean_raw})
    out = Path("reports/_exp_B41_romantic.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
