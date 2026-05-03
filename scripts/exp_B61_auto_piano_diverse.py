"""B61: validate auto_piano on diverse Romantic/Impressionist ASAP composers.
Bach (already known) + Debussy + Brahms. Tests if the median-dur > 0.4 heuristic
correctly switches transcribers for slow chordal pieces."""
from __future__ import annotations
import json, subprocess, pickle
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.audio_io import load_audio
from humscribe.beat.beat_this_track import track_beats_beat_this
from humscribe.config import PipelineConfig
from humscribe.notes import NoteEvent
from humscribe.pipeline import _branch_notes
from humscribe.rhythm.viterbi_quantize import default_allowed_durations, viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, adaptive_pitch_jump, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
SF2 = "/home/swadesh/miniconda3/envs/humscribe/lib/python3.11/site-packages/pretty_midi/TimGM6mb.sf2"
ALLOWED = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TPB = 24

PIECES = [
    "Bach/Fugue/bwv_846",
    "Debussy/Images_Book_1/1_Reflets_dans_lEau",
    "Brahms/Six_Pieces_op_118/2",
]


def render(midi: Path, wav: Path):
    if wav.exists() and wav.stat().st_size > 0: return
    subprocess.run(["fluidsynth", "-ni", "-r", "22050", "-F", str(wav), "-T", "wav", SF2, str(midi)], check=True, capture_output=True)


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
    if not notes: return 0.0, 0
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
    valid = [i for i, n in enumerate(notes) if _hz(n) is not None]
    onsets = np.array([notes[i].onset_s for i in valid])
    offsets = np.array([notes[i].offset_s for i in valid])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([_hz(notes[i]) for i in valid])
    pred_durs = pred_durs[valid]
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, est_p,
                                                  onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if not matched: return 0.0, len(notes)
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap(float(x)) for x in pd])
    gd_s = np.array([snap(float(x)) for x in gd])
    return float(np.mean(pd_s == gd_s)), len(notes)


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B61_auto_piano_diverse",
                     config={"git_sha": git_sha()}, tags=["B61", "asap", "auto_piano"], dir="logs/wandb")
    out_root = Path("/workspace/.cache/asap_renders")
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n  piece                                                    fixed_bd  auto    chosen  med_dur")
    print("  " + "-" * 100)
    for piece_rel in PIECES:
        d = ASAP / piece_rel
        mid = d / "midi_score.mid"
        if not mid.exists(): print(f"  {piece_rel}: no MIDI"); continue
        wav = out_root / (piece_rel.replace("/", "__") + ".wav")
        try: render(mid, wav)
        except Exception as e: print(f"  {piece_rel}: render failed {e}"); continue
        gt_iv, gt_p = load_midi_notes(mid)
        beats, _, _ = track_beats_beat_this(str(wav))
        avg = float(np.diff(beats).mean()) if len(beats) >= 2 else 0.5
        audio, sr = load_audio(str(wav), target_sr=22050)
        cfg_bd = PipelineConfig(input_kind="piano", transcriber="bytedance_piano")
        cfg_auto = PipelineConfig(input_kind="piano", transcriber="auto_piano")
        notes_bd = _branch_notes(str(wav), audio, sr, cfg_bd)
        med_dur = float(np.median([n.offset_s - n.onset_s for n in notes_bd])) if notes_bd else 0.0
        notes_auto = _branch_notes(str(wav), audio, sr, cfg_auto)
        chosen = "bp" if (len(notes_auto) != len(notes_bd) or
                          (notes_auto and notes_bd and notes_auto[0].onset_s != notes_bd[0].onset_s)) else "bd"
        sn_bd, _ = evaluate(notes_bd, beats, gt_iv, gt_p, avg)
        sn_auto, _ = evaluate(notes_auto, beats, gt_iv, gt_p, avg)
        rows.append({"piece": piece_rel, "fixed_bd": sn_bd, "auto_piano": sn_auto, "chosen": chosen, "med_dur": med_dur})
        print(f"  {piece_rel:55s}  {sn_bd:.3f}    {sn_auto:.3f}  {chosen:5s}   {med_dur:.3f}")
        wandb.log({piece_rel: {"fixed_bd": sn_bd, "auto_piano": sn_auto, "med_dur": med_dur}})
    if not rows: run.finish(); return
    means = {k: float(np.mean([r[k] for r in rows])) for k in ("fixed_bd", "auto_piano")}
    print(f"\nMeans:  fixed_bd={means['fixed_bd']:.3f}  auto_piano={means['auto_piano']:.3f}  Δ={means['auto_piano']-means['fixed_bd']:+.3f}")
    wandb.summary.update(means)
    Path("reports/_exp_B61_auto_piano_diverse.json").write_text(json.dumps({"rows": rows, "means": means}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
