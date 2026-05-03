"""B45: HMM segmenter with hybrid PESTO+CREPE voicing. Maybe better-calibrated
voicing helps the HMM."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.hmm_segment import HMMConfig, segment_pitch_to_notes_hmm
from humscribe.pitch.pesto_track import track_pitch_pesto


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def predict(wav: Path, mc: ModeConfig, hmm: HMMConfig):
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, _pv = track_pitch_pesto(audio, sr)
    ct, _ch, cv = track_pitch_crepe(audio, sr)
    voicing = np.interp(pt, ct, cv)
    notes = segment_pitch_to_notes_hmm(pt, ph, voicing, mc, hmm)
    return [n for n in notes if (n.offset_s - n.onset_s) >= mc.min_note_seconds]


def score(notes, iv, hz):
    if not notes: return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, hz, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B45_hmm_with_hybrid",
                      config={"git_sha": git_sha()}, tags=["B45", "vocadito", "hmm"], dir="logs/wandb")
    base = ModeConfig(voicing_threshold=0.75, min_note_seconds=0.052,
                       onset_merge_seconds=0.026, dp_offgrid_penalty=0.5,
                       pitch_smooth_window=19)
    rows = []
    for sigma_v in (0.3, 0.5, 0.7, 1.0):
        for p_start in (0.05, 0.10, 0.20):
            for p_sustain in (0.85, 0.92, 0.97):
                hmm = HMMConfig(p_sustain=p_sustain, p_end=0.05, p_start=p_start,
                                 sigma_voicing=sigma_v, sigma_midi=0.5, interval_decay=0.5)
                f1s = []
                for nf in files:
                    cid = nf.stem.replace("_notesA1", "")
                    wav = audio_dir / f"{cid}.wav"
                    if not wav.exists(): continue
                    gt_iv, gt_p = load_notes(nf)
                    notes = predict(wav, base, hmm)
                    _, _, f = score(notes, gt_iv, gt_p)
                    f1s.append(f)
                mf = float(np.mean(f1s))
                rows.append({"sigma_v": sigma_v, "p_start": p_start, "p_sustain": p_sustain, "f1": mf})
                wandb.log({"sigma_v": sigma_v, "p_start": p_start, "p_sustain": p_sustain, "f1": mf})
    rows.sort(key=lambda r: -r["f1"])
    print(f"\nTop 5 HMM+hybrid:")
    for r in rows[:5]:
        print(f"  F1={r['f1']:.3f}  sigma_v={r['sigma_v']} p_start={r['p_start']} p_sustain={r['p_sustain']}")
    out = Path("reports/_exp_B45_hmm_with_hybrid.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
