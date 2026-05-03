"""B69 — MERT features for BiLSTM voicing/onset detector on Vocadito (Phase C).

Replaces the HuBERT features used in B52 (which underperformed at 0.592) with
MERT (m-a-p/MERT-v1-95M), the music-trained BERT analogue from MAP. MERT was
specifically pre-trained on music with two heads (acoustic + harmonic), so its
embeddings should be more useful for vocal onset detection than HuBERT's
speech-trained ones.

Also adds the "rich" features from B42 alongside MERT (mel-spectrogram +
PESTO + CREPE) to give the BiLSTM the same conditioning that B42 had,
plus the MERT signal.

Pass criterion: Vocadito A1 noff F1 (5-fold CV) ≥ 0.69 (current heuristic 0.665).
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import wandb

from humscribe.audio_io import load_audio
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.notes import midi_to_hz

VOC = Path("~/datasets/vocadito").expanduser()
CACHE = Path("/workspace/.cache/vocadito_mert")
CACHE.mkdir(parents=True, exist_ok=True)
OUT_JSON = Path("reports/_exp_B69_mert_bilstm.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_notes(annotation_csv: Path):
    rows = [r.strip().split(",") for r in annotation_csv.read_text().splitlines() if r.strip()]
    if not rows:
        return np.empty((0, 2)), np.empty(0)
    onsets = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pitches = np.array([float(r[1]) for r in rows], dtype=np.float64)
    durations = np.array([float(r[2]) for r in rows], dtype=np.float64)
    intervals = np.stack([onsets, onsets + durations], axis=1)
    return intervals, pitches


def extract_mert_features(audio: np.ndarray, sr: int, mert_model, mert_proc) -> np.ndarray:
    """Returns (T_frames, D) MERT hidden states resampled to 24kHz."""
    if sr != 24000:
        from scipy.signal import resample_poly
        a = resample_poly(audio, 24000, sr).astype(np.float32)
    else:
        a = audio.astype(np.float32)
    inputs = mert_proc(a, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(mert_model.device) for k, v in inputs.items()}
        out = mert_model(**inputs, output_hidden_states=True)
        # 13 hidden states (1 emb + 12 layers); take avg of last 4 layers
        hs = torch.stack(out.hidden_states[-4:], dim=0).mean(0)  # (1, T, D)
    return hs.squeeze(0).cpu().numpy()


def extract_features(audio_path: str, mert_model, mert_proc) -> dict:
    """Return per-clip cached features. T_frames at MERT's framerate (~75Hz)."""
    cache = CACHE / f"{Path(audio_path).stem}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    audio, sr = load_audio(audio_path, target_sr=22050)
    mert_feat = extract_mert_features(audio, sr, mert_model, mert_proc)  # (T, D)
    # Pitch features at native PESTO rate then resample to MERT timeline
    t_pesto, hz_pesto, vc_pesto = track_pitch_pesto(audio, sr)
    t_crepe, hz_crepe, vc_crepe = track_pitch_crepe(audio, sr)
    n_frames = mert_feat.shape[0]
    # Each MERT frame is ~13.3 ms (24000/320). Build a target timebase.
    mert_t = np.arange(n_frames) * (320.0 / 24000.0)
    # Resample PESTO/CREPE to MERT timebase
    pesto_hz_r = np.interp(mert_t, t_pesto, hz_pesto)
    pesto_vc_r = np.interp(mert_t, t_pesto, vc_pesto)
    crepe_hz_r = np.interp(mert_t, t_crepe, hz_crepe)
    crepe_vc_r = np.interp(mert_t, t_crepe, vc_crepe)
    # Build (T, D+4) feature vector
    aux = np.stack([np.log(np.maximum(pesto_hz_r, 1.0)), pesto_vc_r,
                    np.log(np.maximum(crepe_hz_r, 1.0)), crepe_vc_r], axis=1).astype(np.float32)
    feats = np.concatenate([mert_feat.astype(np.float32), aux], axis=1)  # (T, D+4)
    out = {"feats": feats, "mert_t": mert_t.astype(np.float32),
           "pesto_hz": pesto_hz_r.astype(np.float32),
           "crepe_vc": crepe_vc_r.astype(np.float32)}
    np.savez(cache, **out)
    return out


def build_targets(mert_t: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Per-frame onset-or-active label: 1 if frame is within any [on, off]."""
    t = mert_t
    label = np.zeros(len(t), dtype=np.float32)
    for on, off in intervals:
        mask = (t >= on) & (t <= off)
        label[mask] = 1.0
    return label


class BiLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True,
                             bidirectional=True, dropout=0.2)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)  # (B, T) logits


def voicing_to_intervals(voicing_prob: np.ndarray, t: np.ndarray, threshold: float,
                          min_dur_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Same exit-side hysteresis logic as the heuristic, but with predicted voicing."""
    state = "off"
    intervals = []
    pitches = []
    on_idx = -1
    for i, v in enumerate(voicing_prob):
        if state == "off" and v > threshold:
            state = "on"; on_idx = i
        elif state == "on" and v < threshold:
            state = "off"
            on_t, off_t = t[on_idx], t[i]
            if off_t - on_t >= min_dur_s:
                intervals.append([on_t, off_t])
                pitches.append(0.0)  # filled outside
    if state == "on":
        on_t, off_t = t[on_idx], t[-1]
        if off_t - on_t >= min_dur_s:
            intervals.append([on_t, off_t])
            pitches.append(0.0)
    return np.array(intervals), np.array(pitches)


def assign_pitch(intervals: np.ndarray, t: np.ndarray, pesto_hz: np.ndarray) -> np.ndarray:
    """Median PESTO pitch within each interval."""
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


def main(n_epochs: int = 30, hidden: int = 128, lr: float = 1e-3,
         batch_size: int = 4, threshold: float = 0.5) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "hidden": hidden, "lr": lr,
             "batch_size": batch_size, "threshold": threshold,
             "model": "m-a-p/MERT-v1-95M"}
    run = wandb.init(project="humscribe-v3.2", name="exp_B69_mert_bilstm",
                     config=cfg_w, tags=["B69", "vocadito", "mert", "bilstm", "phase-c"],
                     dir="logs/wandb")

    print("loading MERT-v1-95M")
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    mert_proc = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",
                                                           trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M",
                                            trust_remote_code=True).to("cuda").eval()
    print(f"  mert params: {sum(p.numel() for p in mert_model.parameters())/1e6:.1f}M")

    print("extracting features for all 40 Vocadito clips")
    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    clips = sorted(p.stem for p in audio_dir.glob("vocadito_*.wav"))
    data = []  # list of (clip, feats, target_label, mert_t, ref_intervals_a1, ref_pitches_a1)
    for clip in clips:
        wav = audio_dir / f"{clip}.wav"
        a1 = ann_dir / f"{clip}_notesA1.csv"
        if not a1.exists():
            continue
        try:
            f = extract_features(str(wav), mert_model, mert_proc)
        except Exception as e:
            print(f"  {clip} feature extract failed: {e}")
            continue
        ref_iv, ref_pi = load_notes(a1)
        label = build_targets(f["mert_t"], ref_iv)
        data.append((clip, f["feats"], label, f["mert_t"], f["pesto_hz"], ref_iv, ref_pi))
    print(f"  {len(data)} clips loaded")
    if not data:
        run.finish(); return
    in_dim = data[0][1].shape[1]
    print(f"  feat dim: {in_dim}")
    wandb.summary["feat_dim"] = in_dim
    wandb.summary["n_clips"] = len(data)

    # Free MERT to save VRAM during training
    del mert_model
    torch.cuda.empty_cache()

    # 5-fold CV
    np.random.seed(42)
    perm = np.random.permutation(len(data))
    fold_size = len(data) // 5
    folds = [perm[i*fold_size:(i+1)*fold_size].tolist() for i in range(5)]
    folds[-1].extend(perm[5*fold_size:].tolist())

    f1_per_fold = []
    for fold_idx, val_idx in enumerate(folds):
        val_set = [data[i] for i in val_idx]
        train_set = [data[i] for i in range(len(data)) if i not in val_idx]
        print(f"\nfold {fold_idx}: train={len(train_set)} val={len(val_set)}")
        model = BiLSTM(in_dim=in_dim, hidden=hidden).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        for epoch in range(n_epochs):
            model.train()
            np.random.shuffle(train_set)
            losses = []
            for c, feats, label, *_ in train_set:
                # Truncate to first 30s for memory
                feats_t = torch.from_numpy(feats[:2250]).unsqueeze(0).to("cuda")
                label_t = torch.from_numpy(label[:2250]).unsqueeze(0).to("cuda")
                opt.zero_grad()
                logits = model(feats_t)
                loss = bce(logits, label_t)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            mean_loss = sum(losses) / len(losses)
            wandb.log({"fold": fold_idx, "epoch": epoch, "train_loss": mean_loss})
        # Validate
        model.eval()
        f1s = []
        with torch.no_grad():
            for c, feats, label, mert_t, pesto_hz, ref_iv, ref_pi in val_set:
                feats_t = torch.from_numpy(feats).unsqueeze(0).to("cuda")
                logits = model(feats_t).squeeze(0).cpu().numpy()
                voicing = 1 / (1 + np.exp(-logits))
                pred_iv, _ = voicing_to_intervals(voicing, mert_t, threshold)
                if len(pred_iv) == 0:
                    f1s.append(0.0); continue
                pred_pi = assign_pitch(pred_iv, mert_t, pesto_hz)
                f1 = score_clip(pred_iv, pred_pi, ref_iv, ref_pi)
                f1s.append(f1)
        mean_f1 = float(np.mean(f1s))
        print(f"  fold {fold_idx} val F1: {mean_f1:.4f}")
        f1_per_fold.append(mean_f1)
        wandb.log({"fold": fold_idx, "val_f1": mean_f1})

    overall = float(np.mean(f1_per_fold))
    print(f"\nOVERALL 5-fold CV F1 (no-offset): {overall:.4f}")
    print(f"per-fold: {f1_per_fold}")
    wandb.summary["mean_cv_f1"] = overall
    wandb.summary["cv_std"] = float(np.std(f1_per_fold))
    OUT_JSON.write_text(json.dumps({"f1_per_fold": f1_per_fold, "mean_f1": overall,
                                      "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    main(**vars(ap.parse_args()))
