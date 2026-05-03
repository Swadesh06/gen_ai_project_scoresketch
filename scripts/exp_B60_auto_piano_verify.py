"""B60: end-to-end verification of the new auto_piano transcriber on ASAP Romantic.
Confirms the +9.3pp Chopin gain transfers via the production code path."""
from __future__ import annotations
import json, subprocess, pickle
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig
from humscribe.notes import NoteEvent
from humscribe.pipeline import _branch_notes
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
CACHE_DIR = Path("/workspace/.cache/asap_bytedance")
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"


def render(midi: Path, wav: Path):
    if wav.exists() and wav.stat().st_size > 0: return
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)], check=True, capture_output=True)


def load_cached(piece_dir: Path):
    key = piece_dir.relative_to(ASAP).as_posix().replace("/", "__")
    cache_path = CACHE_DIR / f"{key}.pkl"
    with open(cache_path, "rb") as f:
        d = pickle.load(f)
    return d["beats"]


def load_midi_notes(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv = []; pi = []
    for inst in pm.instruments:
        for n in inst.notes:
            iv.append([n.start, n.end])
            pi.append(440.0 * 2 ** ((n.pitch - 69) / 12))
    return np.array(iv), np.array(pi)


def snap(d): return float(ALLOWED[np.argmin(np.abs(ALLOWED - d))])


def evaluate(notes, beats, gt_iv, gt_p, avg_beat):
    pj = adaptive_pitch_jump(notes)
    cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on, q_off = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=TPB,
                                           allowed_durations_tatums=default_allowed_durations(TPB))
    pred_durs = (q_off - q_on) / TPB
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    def _hz(n):
        if n.pitch_hz: return n.pitch_hz
        if n.pitch_midi is not None: return 440.0 * 2 ** ((n.pitch_midi - 69) / 12)
        return None
    valid_idx = [i for i, n in enumerate(notes) if _hz(n) is not None]
    onsets = np.array([notes[i].onset_s for i in valid_idx])
    offsets = np.array([notes[i].offset_s for i in valid_idx])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([_hz(notes[i]) for i in valid_idx])
    pred_durs = pred_durs[valid_idx]
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, est_p, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if not matched: return 0.0
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    return float(np.mean(pd_s == gd_s))


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B60_auto_piano_verify",
                     config={"git_sha": git_sha()}, tags=["B60", "asap", "auto_piano"], dir="logs/wandb")
    out_root = Path("/workspace/.cache/asap_renders")
    rows = []
    pieces = ["Bach/Fugue/bwv_846", "Beethoven/Piano_Sonatas/21-1",
              "Schumann/Toccata", "Chopin/Berceuse_op_57"]
    print("\n  piece                              fixed_bd  auto_piano  chosen")
    print("  " + "-" * 75)
    for piece_rel in pieces:
        d = ASAP / piece_rel
        cached = CACHE_DIR / (piece_rel.replace("/", "__") + ".pkl")
        mid = d / "midi_score.mid"
        if not (mid.exists() and cached.exists()): continue
        wav = out_root / (piece_rel.replace("/", "__") + ".wav")
        render(mid, wav)
        gt_iv, gt_p = load_midi_notes(mid)
        beats = load_cached(d)
        avg = float(np.diff(beats).mean())
        # Run via _branch_notes for both transcribers
        cfg_bd = PipelineConfig(input_kind="piano", transcriber="bytedance_piano")
        cfg_auto = PipelineConfig(input_kind="piano", transcriber="auto_piano")
        audio, sr = load_audio(str(wav), target_sr=22050)
        notes_bd = _branch_notes(str(wav), audio, sr, cfg_bd)
        notes_auto = _branch_notes(str(wav), audio, sr, cfg_auto)
        chosen = "bd"
        if len(notes_auto) != len(notes_bd) or any(a.onset_s != b.onset_s for a, b in zip(notes_auto[:50], notes_bd[:50])):
            chosen = "bp"
        sn_bd = evaluate(notes_bd, beats, gt_iv, gt_p, avg)
        sn_auto = evaluate(notes_auto, beats, gt_iv, gt_p, avg)
        rows.append({"piece": piece_rel, "fixed_bd": sn_bd, "auto_piano": sn_auto, "chosen": chosen})
        print(f"  {piece_rel:35s}  {sn_bd:.3f}     {sn_auto:.3f}      {chosen}")
        wandb.log({piece_rel: {"fixed_bd": sn_bd, "auto_piano": sn_auto}})
    if not rows: print("nothing"); run.finish(); return
    means = {"fixed_bd": float(np.mean([r["fixed_bd"] for r in rows])),
              "auto_piano": float(np.mean([r["auto_piano"] for r in rows]))}
    print(f"\nMeans:  fixed_bd={means['fixed_bd']:.3f}  auto_piano={means['auto_piano']:.3f}  Δ={means['auto_piano']-means['fixed_bd']:+.3f}")
    wandb.summary.update(means)
    out = Path("reports/_exp_B60_auto_piano_verify.json")
    out.write_text(json.dumps({"rows": rows, "means": means}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
