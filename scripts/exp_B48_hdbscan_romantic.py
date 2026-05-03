"""B48: HDBSCAN voice tracker on Romantic ASAP pieces."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.instrument.piano import transcribe_piano
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_hdbscan import HDBSCANVoiceConfig, assign_voices_hdbscan
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


PIECES = ["Beethoven/Piano_Sonatas/21-1", "Schumann/Toccata",
          "Chopin/Berceuse_op_57", "Liszt/Sonata"]


def eval_piece(piece_dir: Path, voice_method: str, **vt_kwargs):
    mid = piece_dir / "midi_score.mid"
    wav = piece_dir / "midi_score.wav"
    ann = piece_dir / "midi_score_annotations.txt"
    render(mid, wav)
    beats = load_score_beats(ann)
    avg_beat = float(np.diff(beats).mean()) if len(beats) >= 2 else 0.5
    notes = transcribe_piano(str(wav))
    onsets = np.array([n.onset_s for n in notes]); offsets = np.array([n.offset_s for n in notes])
    if voice_method == "greedy":
        cfg = VoiceTrackConfig(**vt_kwargs)
        voices = assign_voices(notes, cfg)
    elif voice_method == "hdbscan":
        cfg = HDBSCANVoiceConfig(**vt_kwargs)
        voices = assign_voices_hdbscan(notes, cfg)
    elif voice_method == "none":
        voices = [list(range(len(notes)))]
    else:
        raise ValueError(voice_method)
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
    if not matched: return 0.0, 0.0, len(voices)
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    raw = float(np.mean(np.abs(pd - gd) < 0.05))
    snap_pct = float(np.mean(pd_s == gd_s))
    return raw, snap_pct, len(voices)


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B48_hdbscan_romantic",
                     config={"git_sha": git_sha()}, tags=["B48", "asap", "hdbscan", "romantic"], dir="logs/wandb")
    rows = []
    configs = [
        ("greedy_default", "greedy", {"pitch_jump": 3.0, "time_gap_s": 0.5}),
        ("greedy_wide", "greedy", {"pitch_jump": 7.0, "time_gap_s": 1.0}),
        ("hdbscan_mcs4", "hdbscan", {"min_cluster_size": 4, "min_samples": 2}),
        ("hdbscan_mcs6", "hdbscan", {"min_cluster_size": 6, "min_samples": 2}),
        ("hdbscan_mcs10", "hdbscan", {"min_cluster_size": 10, "min_samples": 3}),
        ("no_vt", "none", {}),
    ]
    for name, method, kw in configs:
        per = []
        for piece_rel in PIECES:
            d = ASAP / piece_rel
            if not (d / "midi_score.mid").exists():
                continue
            try:
                raw, snap_pct, n_voices = eval_piece(d, method, **kw)
            except Exception as e:
                print(f"  {piece_rel}: failed -- {e}"); continue
            per.append({"piece": piece_rel, "raw": raw, "snap": snap_pct, "n_voices": n_voices})
        if not per: continue
        mean_snap = float(np.mean([r["snap"] for r in per]))
        mean_raw = float(np.mean([r["raw"] for r in per]))
        rows.append({"config": name, "method": method, "kw": kw,
                     "mean_snap": mean_snap, "mean_raw": mean_raw, "per": per})
        print(f"{name:18s}  mean_snap={mean_snap:.3f}  mean_raw={mean_raw:.3f}")
        for r in per:
            print(f"   {r['piece']:35s}  snap={r['snap']:.3f}  voices={r['n_voices']}")
        wandb.log({f"{name}/mean_snap": mean_snap, f"{name}/mean_raw": mean_raw})
    out = Path("reports/_exp_B48_hdbscan_romantic.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nrun: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
