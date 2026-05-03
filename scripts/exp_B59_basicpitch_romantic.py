"""B59: basic_pitch on Romantic ASAP — see if a different transcriber helps.
ByteDance loses 18.8pp on Romantic. Compare basic_pitch as alternative."""
from __future__ import annotations
import json, subprocess, pickle
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
CACHE_DIR = Path("/workspace/.cache/asap_bytedance")
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
ALLOWED_BEATS = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24


def render(midi: Path, wav: Path):
    if wav.exists() and wav.stat().st_size > 0: return
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)], check=True, capture_output=True)


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
        if hz is None or midi is None: continue
        notes.append(NoteEvent(onset_s=x["on"], offset_s=x["off"], pitch_midi=midi, pitch_hz=hz, velocity=x.get("vel", 80), confidence=x.get("conf", 1.0)))
    return notes, d["beats"]


def basic_pitch_transcribe(wav: Path) -> list[NoteEvent]:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, midi_data, _ = predict(str(wav), model_or_model_path=ICASSP_2022_MODEL_PATH)
    out = []
    for inst in midi_data.instruments:
        for n in inst.notes:
            hz = 440.0 * 2 ** ((n.pitch - 69) / 12)
            out.append(NoteEvent(onset_s=n.start, offset_s=n.end, pitch_midi=n.pitch, pitch_hz=hz, velocity=n.velocity, confidence=1.0))
    out.sort(key=lambda e: e.onset_s)
    return out


def load_midi_notes(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv = []; pi = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv), np.array(pi)


def snap(d): return float(ALLOWED_BEATS[np.argmin(np.abs(ALLOWED_BEATS - d))])


def evaluate_pipeline(notes, beats, gt_iv, gt_p, avg_beat):
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
    if not matched: return 0.0, 0
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    return float(np.mean(pd_s == gd_s)), len(matched)


PIECES = ["Bach/Fugue/bwv_846", "Beethoven/Piano_Sonatas/21-1",
          "Schumann/Toccata", "Chopin/Berceuse_op_57"]


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B59_basicpitch_romantic",
                     config={"git_sha": git_sha()}, tags=["B59", "asap", "basicpitch"], dir="logs/wandb")
    rows = []
    print("\n  piece                              ByteDance  basic_pitch  bd_n  bp_n")
    print("  " + "-" * 80)
    out_root = Path("/workspace/.cache/asap_renders")
    out_root.mkdir(parents=True, exist_ok=True)
    for piece_rel in PIECES:
        d = ASAP / piece_rel
        cached = CACHE_DIR / (piece_rel.replace("/", "__") + ".pkl")
        mid = d / "midi_score.mid"
        if not (mid.exists() and cached.exists()):
            print(f"  {piece_rel}: missing"); continue
        wav = out_root / (piece_rel.replace("/", "__") + ".wav")
        try: render(mid, wav)
        except Exception as e: print(f"  {piece_rel}: render failed {e}"); continue
        gt_iv, gt_p = load_midi_notes(mid)
        bd_notes, bd_beats = load_cached(d)
        bd_avg = float(np.diff(bd_beats).mean())
        sn_bd, n_bd = evaluate_pipeline(bd_notes, bd_beats, gt_iv, gt_p, bd_avg)
        try:
            bp_notes = basic_pitch_transcribe(wav)
            sn_bp, n_bp = evaluate_pipeline(bp_notes, bd_beats, gt_iv, gt_p, bd_avg)
        except Exception as e:
            print(f"  {piece_rel}: basic_pitch failed {e}"); continue
        rows.append({"piece": piece_rel, "bd_snap": sn_bd, "bp_snap": sn_bp, "bd_n": n_bd, "bp_n": n_bp})
        print(f"  {piece_rel:35s}  {sn_bd:.3f}      {sn_bp:.3f}        {n_bd:5d} {n_bp:5d}")
        wandb.log({piece_rel: {"bd": sn_bd, "bp": sn_bp}})
    if not rows:
        print("nothing"); run.finish(); return
    means = {"bd_snap": float(np.mean([r["bd_snap"] for r in rows])),
              "bp_snap": float(np.mean([r["bp_snap"] for r in rows]))}
    print(f"\nMeans:  bd={means['bd_snap']:.3f}  bp={means['bp_snap']:.3f}  Δ={means['bp_snap']-means['bd_snap']:+.3f}")
    wandb.summary.update(means)
    out = Path("reports/_exp_B59_basicpitch_romantic.json")
    out.write_text(json.dumps({"rows": rows, "means": means}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
