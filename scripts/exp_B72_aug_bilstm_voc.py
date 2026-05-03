"""B72 — Aggressive augmentation BiLSTM on Vocadito + MTG-QBH (Phase D).

Builds on B70 (Vocadito + MTG-QBH pseudo-labels). The B70 v2 result showed
combined > voconly by ~5-8pp per fold, but absolute F1 still ~0.4 vs
heuristic 0.665. Hypothesis: aggressive online data augmentation could
push the BiLSTM past the heuristic. Not subject to the "data quantity
ceiling" the same way because each augmented sample is a "new" example.

Augmentations applied per training step:
- Pitch shift ±2 semitones via PESTO/CREPE feature shift
- Time stretch 0.85-1.15x via feature dilation
- Noise injection on aux features
- Frequency masking on mel features (SpecAugment-style)

Long training: 80 epochs, hidden=256, 5-fold CV. Designed to take 1-2 hours.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import wandb

from humscribe.audio_io import load_audio
from humscribe.config import ModeConfig
from humscribe.notes import midi_to_hz
from humscribe.pitch.ensemble import track_pitch_hybrid_voicing
from humscribe.pitch.voicing import segment_pitch_to_notes

VOC = Path("~/datasets/vocadito").expanduser()
QBH = Path("~/datasets/mtg_qbh/audio").expanduser()
CACHE = Path("/workspace/.cache/voc_qbh_features")
OUT_JSON = Path("reports/_exp_B72_aug_bilstm.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_vocadito_notes(annotation_csv: Path):
    rows = [r.strip().split(",") for r in annotation_csv.read_text().splitlines() if r.strip()]
    if not rows:
        return np.empty((0, 2)), np.empty(0)
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([onsets, onsets + durations], axis=1), pitches


def pseudo_label_clip(audio_path: str):
    audio, sr = load_audio(audio_path, target_sr=22050)
    t, hz, vc = track_pitch_hybrid_voicing(audio, sr)
    mode_cfg = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    notes = segment_pitch_to_notes(t, hz, vc, mode_cfg)
    notes = [n for n in notes if (n.offset_s - n.onset_s) >= mode_cfg.min_note_seconds]
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    pitches = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    return intervals, pitches


def build_targets(t: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    label = np.zeros(len(t), dtype=np.float32)
    for on, off in intervals:
        label[(t >= on) & (t <= off)] = 1.0
    return label


def load_or_extract(audio_path: str):
    """Reuse B70 cache (mel + PESTO + CREPE on 100Hz timebase)."""
    cache = CACHE / f"{Path(audio_path).stem}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    raise FileNotFoundError(f"feature cache missing for {audio_path}; run B70 first")


def augment(feats: np.ndarray, label: np.ndarray, rng: np.random.RandomState):
    """Apply pitch shift / time stretch / noise / spec masking."""
    f = feats.copy()
    # Pitch shift: features 64..67 are log(pesto_hz), pesto_vc, log(crepe_hz), crepe_vc
    # Shift the log-frequency by random ±0.16 (= ±2 semitones in log2 -> log_e * (semitones/12 * ln(2)))
    shift_st = rng.uniform(-2.0, 2.0)
    log_shift = shift_st * np.log(2) / 12
    f[:, 64] += log_shift  # log(pesto_hz)
    f[:, 66] += log_shift  # log(crepe_hz)
    # Time stretch: dilate features by random factor
    stretch = rng.uniform(0.85, 1.15)
    new_T = max(int(len(f) / stretch), 8)
    if new_T != len(f):
        idx = np.linspace(0, len(f) - 1, new_T).astype(np.int32)
        f = f[idx]
        label = label[idx]
    # Gaussian noise on aux features (last 4 columns)
    f[:, 64:] += rng.normal(0, 0.05, size=f[:, 64:].shape).astype(np.float32)
    # SpecAugment: mask 1-3 random time chunks of 10-20 frames
    n_masks = rng.randint(1, 4)
    for _ in range(n_masks):
        if len(f) > 30:
            t0 = rng.randint(0, len(f) - 20)
            tlen = rng.randint(5, 20)
            f[t0:t0+tlen, :64] = 0  # mask mel only, keep pitch features
    return f, label


class BiLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                             bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def voicing_to_intervals(voicing_prob: np.ndarray, t: np.ndarray, threshold: float,
                          min_dur_s: float = 0.05):
    state = "off"
    intervals = []
    on_idx = -1
    for i, v in enumerate(voicing_prob):
        if state == "off" and v > threshold:
            state = "on"; on_idx = i
        elif state == "on" and v < threshold:
            state = "off"
            on_t, off_t = t[on_idx], t[i]
            if off_t - on_t >= min_dur_s:
                intervals.append([on_t, off_t])
    if state == "on":
        on_t, off_t = t[on_idx], t[-1]
        if off_t - on_t >= min_dur_s:
            intervals.append([on_t, off_t])
    return np.array(intervals)


def assign_pitch(intervals, t, pesto_hz):
    out = []
    for on, off in intervals:
        mask = (t >= on) & (t <= off) & (pesto_hz > 0)
        if mask.sum() > 0:
            out.append(float(np.median(pesto_hz[mask])))
        else:
            out.append(220.0)
    return np.array(out)


def score_clip(intervals_pred, pitches_pred, intervals_ref, pitches_ref):
    import mir_eval
    if len(intervals_pred) == 0 or len(intervals_ref) == 0:
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        intervals_ref, pitches_ref, intervals_pred, pitches_pred,
        offset_ratio=None, onset_tolerance=0.05,
    )[2]


def main(n_epochs: int = 80, hidden: int = 256, lr: float = 1e-3,
         threshold: float = 0.5, n_aug_per_clip: int = 4) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "hidden": hidden,
             "lr": lr, "threshold": threshold, "n_aug_per_clip": n_aug_per_clip}
    run = wandb.init(project="humscribe-v3.2", name="exp_B72_aug_bilstm",
                     config=cfg_w, tags=["B72", "vocadito", "augmentation",
                                          "bilstm", "phase-d"],
                     dir="logs/wandb")

    print("loading cached features (extracted by B70)")
    qbh_data = []
    for wav in sorted(QBH.glob("*.wav")):
        try:
            f = load_or_extract(str(wav))
            iv, _ = pseudo_label_clip(str(wav))
            label = build_targets(f["t"], iv)
            qbh_data.append((wav.stem, f["feats"], label))
        except FileNotFoundError:
            continue
    print(f"  qbh cached: {len(qbh_data)}")
    voc_data = []
    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    for wav in sorted(audio_dir.glob("vocadito_*.wav")):
        a1 = ann_dir / f"{wav.stem}_notesA1.csv"
        if not a1.exists(): continue
        try:
            f = load_or_extract(str(wav))
        except FileNotFoundError:
            continue
        ref_iv, ref_pi = load_vocadito_notes(a1)
        label = build_targets(f["t"], ref_iv)
        voc_data.append((wav.stem, f["feats"], label, f["t"], f["pesto_hz"], ref_iv, ref_pi))
    print(f"  voc cached: {len(voc_data)}")
    in_dim = voc_data[0][1].shape[1]
    print(f"  feat dim: {in_dim}")

    np.random.seed(42)
    perm = np.random.permutation(len(voc_data))
    fold_size = len(voc_data) // 5
    folds = [perm[i*fold_size:(i+1)*fold_size].tolist() for i in range(5)]
    folds[-1].extend(perm[5*fold_size:].tolist())

    f1_per_fold = []
    for fold_idx, val_idx in enumerate(folds):
        val_set = [voc_data[i] for i in val_idx]
        voc_train = [voc_data[i] for i in range(len(voc_data)) if i not in val_idx]
        combined = []
        for c, f_, lbl, *_ in voc_train:
            combined.append((c, f_, lbl))
        for c, f_, lbl in qbh_data:
            combined.append((c, f_, lbl))
        print(f"\n=== fold {fold_idx} ===")
        print(f"  base train: {len(combined)} (with {n_aug_per_clip}x aug = {len(combined)*n_aug_per_clip} effective)")

        model = BiLSTM(in_dim=in_dim, hidden=hidden).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.RandomState(fold_idx * 100)
        best_f1 = 0.0

        for epoch in range(n_epochs):
            model.train()
            losses = []
            np.random.shuffle(combined)
            for c, feats, label in combined:
                # 1 original + n_aug_per_clip augmented
                samples = [(feats, label)]
                for _ in range(n_aug_per_clip):
                    samples.append(augment(feats, label, rng))
                for f_aug, lbl_aug in samples:
                    feats_t = torch.from_numpy(f_aug[:3000]).unsqueeze(0).to("cuda")
                    label_t = torch.from_numpy(lbl_aug[:3000]).unsqueeze(0).to("cuda")
                    opt.zero_grad()
                    logits = model(feats_t)
                    loss = bce(logits, label_t)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
            sched.step()
            mean_loss = sum(losses) / len(losses)
            wandb.log({"fold": fold_idx, "epoch": epoch, "train_loss": mean_loss,
                        "lr": sched.get_last_lr()[0]})

            # Eval every 10 epochs and at end
            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                model.eval()
                f1s = []
                with torch.no_grad():
                    for c, feats, label, t_axis, pesto_hz, ref_iv, ref_pi in val_set:
                        feats_t = torch.from_numpy(feats).unsqueeze(0).to("cuda")
                        logits = model(feats_t).squeeze(0).cpu().numpy()
                        voicing = 1 / (1 + np.exp(-logits))
                        pred_iv = voicing_to_intervals(voicing, t_axis, threshold)
                        if len(pred_iv) == 0:
                            f1s.append(0.0); continue
                        pred_pi = assign_pitch(pred_iv, t_axis, pesto_hz)
                        f1 = score_clip(pred_iv, pred_pi, ref_iv, ref_pi)
                        f1s.append(f1)
                mean_f1 = float(np.mean(f1s))
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                print(f"  fold {fold_idx} epoch {epoch:3d} loss={mean_loss:.3f} val_F1={mean_f1:.4f} (best {best_f1:.4f})")
                wandb.log({"fold": fold_idx, "epoch": epoch, "val_f1": mean_f1,
                           "best_f1_so_far": best_f1})

        f1_per_fold.append(best_f1)

    overall = float(np.mean(f1_per_fold))
    print(f"\n5-fold CV mean (best_per_fold) F1: {overall:.4f}")
    wandb.summary["mean_cv_f1"] = overall
    wandb.summary["cv_std"] = float(np.std(f1_per_fold))
    OUT_JSON.write_text(json.dumps({"f1_per_fold": f1_per_fold, "mean_f1": overall,
                                     "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-aug-per-clip", type=int, default=4)
    main(**vars(ap.parse_args()))
