"""PESTO sanity check on its own training distribution.
Gate: Raw Pitch Accuracy > 0.85 across 5 random MIR-1K clips.
If this fails, the bug is in *our* loading/voicing - not PESTO.
"""
import argparse, random
from pathlib import Path
import mir_eval, numpy as np, soundfile as sf
from humscribe.pitch.pesto_track import track_pitch_pesto

def load_pv(pv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """MIR-1K pitch labels: one MIDI semitone (or 0=unvoiced) per 20 ms."""
    midi = np.array([float(x) for x in pv_path.read_text().split()])
    times = np.arange(len(midi)) * 0.020 + 0.020   # 20 ms hop, 20 ms offset
    return times, midi

def main(mir1k_dir, n_clips):
    root = Path(mir1k_dir).expanduser()
    wavs = list((root / "Wavfile").glob("*.wav"))
    random.seed(0)
    sample = random.sample(wavs, n_clips)
    rpas = []
    for wav in sample:
        audio, sr = sf.read(str(wav))
        if audio.ndim == 2:                     # MIR-1K is stereo: L=accomp, R=vocal
            audio = audio[:, 1]
        gt_t, gt_midi = load_pv(root / "PitchLabel" / wav.with_suffix(".pv").name)
        pred_t, pred_hz, _ = track_pitch_pesto(audio.astype(np.float32), sr)
        pred_cents = 1200 * np.log2(pred_hz / 440.0 + 1e-9) + 6900   # to MIDI*100
        # interpolate to GT timestamps
        pred_at_gt = np.interp(gt_t, pred_t, pred_cents)
        gt_voicing = (gt_midi > 0).astype(bool)
        gt_cents = np.where(gt_voicing, gt_midi * 100, 0.0)
        rpa = mir_eval.melody.raw_pitch_accuracy(
            gt_voicing, gt_cents, gt_voicing, pred_at_gt,
            cent_tolerance=50,
        )
        print(f"{wav.name:30s}  RPA={rpa:.3f}")
        rpas.append(rpa)
    mean = float(np.mean(rpas))
    print(f"\nMean RPA: {mean:.3f}  N={len(rpas)}")
    print(f"GATE: {'PASS' if mean > 0.85 else 'FAIL - fix loading/voicing, not PESTO'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mir1k-dir", default="~/datasets/mir1k")
    ap.add_argument("--n-clips", type=int, default=5)
    main(**vars(ap.parse_args()))
