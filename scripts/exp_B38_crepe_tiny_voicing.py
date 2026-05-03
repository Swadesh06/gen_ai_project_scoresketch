"""B38: PESTO pitch + CREPE-tiny voicing (faster than CREPE-full). Compare F1."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb, librosa, torch, torchcrepe

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.voicing import segment_pitch_to_notes


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def crepe_voicing(audio: np.ndarray, sr: int, model: str) -> tuple[np.ndarray, np.ndarray]:
    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    x = torch.from_numpy(audio).unsqueeze(0).cuda()
    hop = 160
    _, periodicity = torchcrepe.predict(x, sr, hop_length=hop, fmin=50, fmax=1100,
                                         model=model, return_periodicity=True,
                                         batch_size=512, device="cuda")
    cv = periodicity.squeeze(0).cpu().numpy().astype(np.float64)
    times = np.arange(len(cv)) * (hop / sr)
    return times, cv


def predict(wav: Path, mc: ModeConfig, model: str) -> list[NoteEvent]:
    audio, sr = load_audio(str(wav), target_sr=22050)
    pt, ph, _pv = track_pitch_pesto(audio, sr)
    ct, cv = crepe_voicing(audio, sr, model)
    voicing = np.interp(pt, ct, cv)
    notes = segment_pitch_to_notes(pt, ph, voicing, mc)
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
    run = wandb.init(project="humscribe-v3.2", name="exp_B38_crepe_tiny_voicing",
                     config={"git_sha": git_sha()}, tags=["B38", "vocadito", "tiny"], dir="logs/wandb")
    for model in ("tiny", "small", "medium", "full"):
        # use B36b best vt+psw (0.75, 19) as baseline
        mc = ModeConfig(voicing_threshold=0.75, min_note_seconds=0.052,
                        onset_merge_seconds=0.026, dp_offgrid_penalty=0.5,
                        pitch_smooth_window=19)
        f1s = []
        import time; t0 = time.time()
        for nf in files:
            cid = nf.stem.replace("_notesA1", "")
            wav = audio_dir / f"{cid}.wav"
            if not wav.exists(): continue
            gt_iv, gt_p = load_notes(nf)
            try:
                notes = predict(wav, mc, model)
            except Exception as e:
                print(f"{cid}: {e}"); continue
            _, _, f = score(notes, gt_iv, gt_p)
            f1s.append(f)
        mf = float(np.mean(f1s)) if f1s else 0.0
        dt = time.time() - t0
        print(f"  crepe-{model}  F1={mf:.3f}  wall={dt:.0f}s")
        wandb.log({"model": model, "f1": mf, "wall_s": dt})
    run.finish()


if __name__ == "__main__":
    main()
