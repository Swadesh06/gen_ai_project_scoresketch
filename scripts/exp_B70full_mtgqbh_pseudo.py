"""B70 — Item 5 (substitute): MTG-QBH pseudo-label + BiLSTM training (Phase C).

The original v2-spec item 5 wanted MedleyDB-Melody (108 clips). MedleyDB
requires registration. MTG-QBH (118 clips bootstrapped, all humming) is the
natural substitute:
- Same domain (humming, monophonic vocal)
- Available on Zenodo without registration
- Has no f0 annotations → must pseudo-label

Plan per v2 §item 5:
1. Pseudo-label MTG-QBH onsets/durations using the heuristic voicing+pitch
   pipeline (same one currently in production at Vocadito A1 noff F1=0.665).
2. Combine Vocadito (40, real labels) + MTG-QBH (118, pseudo labels) → 158 clips.
3. Train BiLSTM with mel + PESTO + CREPE features (B42 recipe).
4. 5-fold CV on the Vocadito subset only (for an honest comparison).

Pass criterion: Vocadito A1 noff F1 (5-fold CV) ≥ 0.69.
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
CACHE.mkdir(parents=True, exist_ok=True)
OUT_JSON = Path("reports/_exp_B70full_mtgqbh_pseudo.json")


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


def pseudo_label_clip(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the heuristic voicing+pitch pipeline; convert resulting NoteEvents to (intervals, pitches)."""
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


def extract_features(audio_path: str, target_hop_s: float = 0.01) -> dict:
    """Mel + PESTO + CREPE concatenated, all on a common 100Hz timebase."""
    cache = CACHE / f"{Path(audio_path).stem}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    audio, sr = load_audio(audio_path, target_sr=22050)
    # Mel-spectrogram at 100Hz
    hop_length = int(round(sr * target_hop_s))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048,
                                           hop_length=hop_length, n_mels=64)
    mel_db = librosa.power_to_db(mel).T  # (T, 64)
    # PESTO + CREPE on the same audio
    from humscribe.pitch.pesto_track import track_pitch_pesto
    from humscribe.pitch.crepe_track import track_pitch_crepe
    t_pesto, hz_pesto, vc_pesto = track_pitch_pesto(audio, sr)
    t_crepe, hz_crepe, vc_crepe = track_pitch_crepe(audio, sr)
    n_frames = mel_db.shape[0]
    target_t = np.arange(n_frames) * target_hop_s
    pesto_hz_r = np.interp(target_t, t_pesto, hz_pesto)
    pesto_vc_r = np.interp(target_t, t_pesto, vc_pesto)
    crepe_hz_r = np.interp(target_t, t_crepe, hz_crepe)
    crepe_vc_r = np.interp(target_t, t_crepe, vc_crepe)
    aux = np.stack([np.log(np.maximum(pesto_hz_r, 1.0)), pesto_vc_r,
                    np.log(np.maximum(crepe_hz_r, 1.0)), crepe_vc_r], axis=1)
    feats = np.concatenate([mel_db, aux], axis=1).astype(np.float32)  # (T, 68)
    out = {"feats": feats, "t": target_t.astype(np.float32),
           "pesto_hz": pesto_hz_r.astype(np.float32)}
    np.savez(cache, **out)
    return out


def build_targets(t: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    label = np.zeros(len(t), dtype=np.float32)
    for on, off in intervals:
        label[(t >= on) & (t <= off)] = 1.0
    return label


class BiLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                             bidirectional=True, dropout=0.2)
        self.head = nn.Linear(hidden * 2, 1)

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


def assign_pitch(intervals: np.ndarray, t: np.ndarray, pesto_hz: np.ndarray) -> np.ndarray:
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


def main(n_epochs: int = 40, hidden: int = 192, lr: float = 1e-3,
         threshold: float = 0.5) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "hidden": hidden,
             "lr": lr, "threshold": threshold}
    run = wandb.init(project="humscribe-v3.2", name="exp_B70full_mtgqbh_pseudo",
                     config=cfg_w, tags=["B70", "vocadito", "mtg-qbh", "pseudo-label",
                                          "phase-c", "item5"],
                     dir="logs/wandb")

    print("STEP 1: extract features + pseudo-label MTG-QBH")
    qbh_paths = sorted(QBH.glob("*.wav"))
    print(f"  MTG-QBH clips: {len(qbh_paths)}")
    qbh_data = []
    for i, wav in enumerate(qbh_paths):
        try:
            f = extract_features(str(wav))
            iv, _ = pseudo_label_clip(str(wav))
            label = build_targets(f["t"], iv)
            qbh_data.append((wav.stem, f["feats"], label, f["t"], f["pesto_hz"]))
        except Exception as e:
            print(f"  {wav.stem} failed: {e}")
        if (i+1) % 20 == 0:
            print(f"  ...{i+1}/{len(qbh_paths)} done")
    print(f"  pseudo-labeled: {len(qbh_data)}")

    print("\nSTEP 2: extract features + load real labels for Vocadito")
    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    voc_data = []
    for wav in sorted(audio_dir.glob("vocadito_*.wav")):
        a1 = ann_dir / f"{wav.stem}_notesA1.csv"
        if not a1.exists():
            continue
        f = extract_features(str(wav))
        ref_iv, ref_pi = load_vocadito_notes(a1)
        label = build_targets(f["t"], ref_iv)
        voc_data.append((wav.stem, f["feats"], label, f["t"], f["pesto_hz"], ref_iv, ref_pi))
    print(f"  vocadito: {len(voc_data)}")
    in_dim = voc_data[0][1].shape[1]
    print(f"  feat dim: {in_dim}")

    np.random.seed(42)
    perm = np.random.permutation(len(voc_data))
    fold_size = len(voc_data) // 5
    folds = [perm[i*fold_size:(i+1)*fold_size].tolist() for i in range(5)]
    folds[-1].extend(perm[5*fold_size:].tolist())

    f1_per_fold_combined = []
    f1_per_fold_voconly = []
    for fold_idx, val_idx in enumerate(folds):
        val_set = [voc_data[i] for i in val_idx]
        voc_train = [voc_data[i] for i in range(len(voc_data)) if i not in val_idx]

        print(f"\n=== fold {fold_idx} ===")
        print(f"  voc-train: {len(voc_train)}; voc-val: {len(val_set)}; qbh-extra: {len(qbh_data)}")

        # Train on Vocadito + MTG-QBH pseudo
        combined_train = []
        for c, f_, lbl, *_ in voc_train:
            combined_train.append((c, f_, lbl))
        for c, f_, lbl, *_ in qbh_data:
            combined_train.append((c, f_, lbl))

        # Two heads: combined and vocadito-only baseline
        for label_kind, train_set in [("combined", combined_train),
                                       ("voconly", [(c, f_, lbl) for c, f_, lbl, *_ in voc_train])]:
            model = BiLSTM(in_dim=in_dim, hidden=hidden).to("cuda")
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            bce = nn.BCEWithLogitsLoss()
            for epoch in range(n_epochs):
                model.train()
                losses = []
                np.random.shuffle(train_set)
                for c, feats, label in train_set:
                    feats_t = torch.from_numpy(feats[:3000]).unsqueeze(0).to("cuda")
                    label_t = torch.from_numpy(label[:3000]).unsqueeze(0).to("cuda")
                    opt.zero_grad()
                    logits = model(feats_t)
                    loss = bce(logits, label_t)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                wandb.log({"fold": fold_idx, f"{label_kind}_train_loss": sum(losses)/len(losses),
                            "epoch": epoch})

            # Validate on Vocadito real labels
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
            print(f"  fold {fold_idx} {label_kind} val F1: {mean_f1:.4f}")
            wandb.log({"fold": fold_idx, f"{label_kind}_val_f1": mean_f1})
            if label_kind == "combined":
                f1_per_fold_combined.append(mean_f1)
            else:
                f1_per_fold_voconly.append(mean_f1)

    overall_combined = float(np.mean(f1_per_fold_combined))
    overall_voconly = float(np.mean(f1_per_fold_voconly))
    delta = overall_combined - overall_voconly
    print(f"\n5-fold CV mean F1 (combined  voc + qbh-pseudo): {overall_combined:.4f}")
    print(f"5-fold CV mean F1 (vocadito only):              {overall_voconly:.4f}")
    print(f"delta (pseudo-label gain):                      {delta:+.4f}")
    wandb.summary.update({
        "mean_f1_combined": overall_combined, "mean_f1_voconly": overall_voconly,
        "pseudo_gain": delta,
    })
    OUT_JSON.write_text(json.dumps({
        "f1_per_fold_combined": f1_per_fold_combined,
        "f1_per_fold_voconly": f1_per_fold_voconly,
        "mean_f1_combined": overall_combined,
        "mean_f1_voconly": overall_voconly,
        "pseudo_gain": delta,
        "config": cfg_w,
    }, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    main(**vars(ap.parse_args()))
