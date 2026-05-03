"""B85 — Learned offset corrector for Vocadito humming (Phase D).

Targets the unfixed Vocadito offset20 gap: heuristic F1 = 0.439 vs IAA = 0.642.
B47 (entry hysteresis) and B62 (exit hysteresis) didn't help. B56 (DP duration
snapping) didn't help.

This model takes the heuristic's predicted (onset, offset_initial) plus a
local voicing/pitch context window and predicts a corrected offset.

Per-note inputs (each → 33-dim feature vector):
- log(predicted_duration_initial)
- onset position in clip (0-1)
- voicing trace 30 frames around offset (±0.15s)
- pitch trace (log-Hz) 30 frames around offset

Output: corrected_offset_delta (signed seconds, target = gt_offset - predicted_offset).

Training: pair each predicted note with its mir_eval-matched GT note for
the corrected_offset target. Skip notes without a matched GT.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

import mir_eval
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
CACHE = Path("/workspace/.cache/voc_offset_corrector")
CACHE.mkdir(parents=True, exist_ok=True)
OUT_JSON = Path("reports/_exp_B85_offset_corrector.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_vocadito_notes(p):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    if not rows: return np.empty((0, 2)), np.empty(0)
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def predict_clip(audio_path: str):
    cache_p = CACHE / f"pred_{Path(audio_path).stem}.npz"
    if cache_p.exists():
        d = np.load(cache_p, allow_pickle=False)
        return d["intervals"], d["pitches"], d["pesto_t"], d["pesto_hz"], d["pesto_vc"]
    audio, sr = load_audio(audio_path, target_sr=22050)
    t, hz, vc = track_pitch_hybrid_voicing(audio, sr)
    mode_cfg = ModeConfig.for_mode("soft", pitch_model="pesto_crepevoicing")
    notes = segment_pitch_to_notes(t, hz, vc, mode_cfg)
    notes = [n for n in notes if (n.offset_s - n.onset_s) >= mode_cfg.min_note_seconds]
    intervals = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    pitches = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    np.savez(cache_p, intervals=intervals, pitches=pitches,
             pesto_t=t, pesto_hz=hz, pesto_vc=vc)
    return intervals, pitches, t, hz, vc


def build_features(pred_iv, pred_pi, pesto_t, pesto_hz, pesto_vc, frames_pre=15, frames_post=15):
    """For each predicted note, build a feature vector around its predicted offset."""
    n = len(pred_iv)
    if n == 0:
        return np.empty((0, 32))
    feats = np.zeros((n, 32), dtype=np.float32)
    clip_dur = float(pred_iv[-1, 1])
    hop = pesto_t[1] - pesto_t[0] if len(pesto_t) > 1 else 0.01
    for i in range(n):
        on, off = pred_iv[i]
        feats[i, 0] = np.log(max(off - on, 1e-3))
        feats[i, 1] = on / max(clip_dur, 1e-3)
        # Voicing + pitch trace around offset
        idx_off = int(round(off / hop))
        for k in range(-frames_pre, frames_post):
            j = idx_off + k
            if 0 <= j < len(pesto_t):
                feats[i, 2 + (k + frames_pre)] = pesto_vc[j]
    return feats


def match_to_gt(pred_iv, pred_pi, gt_iv, gt_pi):
    if len(pred_iv) == 0 or len(gt_iv) == 0:
        return []
    return mir_eval.transcription.match_notes(
        gt_iv, gt_pi, pred_iv, pred_pi,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )


class OffsetMLP(nn.Module):
    def __init__(self, in_dim=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),  # corrected_offset_delta
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def score_offset_f1(pred_iv, pred_pi, gt_iv, gt_pi, ratio=0.2):
    if len(pred_iv) == 0 or len(gt_iv) == 0:
        return 0.0
    return mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_pi, pred_iv, pred_pi,
        offset_ratio=ratio, onset_tolerance=0.05, pitch_tolerance=50.0,
    )[2]


def main(n_epochs: int = 100, hidden: int = 128, lr: float = 1e-3) -> None:
    cfg_w = {"git_sha": git_sha(), "n_epochs": n_epochs, "hidden": hidden, "lr": lr}
    run = wandb.init(project="humscribe-v3.2", name="exp_B85_offset_corrector",
                     config=cfg_w, tags=["B85", "vocadito", "offset", "phase-d"],
                     dir="logs/wandb")

    audio_dir = VOC / "Audio"
    ann_dir = VOC / "Annotations" / "Notes"
    clips = sorted(p.stem for p in audio_dir.glob("vocadito_*.wav"))
    print(f"loading {len(clips)} clips")
    all_data = []
    for c in clips:
        wav = audio_dir / f"{c}.wav"
        a1 = ann_dir / f"{c}_notesA1.csv"
        if not a1.exists(): continue
        try:
            pred_iv, pred_pi, pt, ph, pv = predict_clip(str(wav))
        except Exception as e:
            print(f"  {c} failed: {e}")
            continue
        gt_iv, gt_pi = load_vocadito_notes(a1)
        feats = build_features(pred_iv, pred_pi, pt, ph, pv)
        matched = match_to_gt(pred_iv, pred_pi, gt_iv, gt_pi)
        # For each match, target = gt_offset - pred_offset
        target = np.zeros(len(pred_iv), dtype=np.float32)
        mask = np.zeros(len(pred_iv), dtype=bool)
        for gi, pi in matched:
            target[pi] = float(gt_iv[gi, 1] - pred_iv[pi, 1])
            mask[pi] = True
        all_data.append({
            "clip": c, "pred_iv": pred_iv, "pred_pi": pred_pi,
            "gt_iv": gt_iv, "gt_pi": gt_pi,
            "feats": feats, "target": target, "mask": mask,
        })
    print(f"  loaded {len(all_data)}")

    np.random.seed(42)
    perm = np.random.permutation(len(all_data))
    fold_size = len(all_data) // 5
    folds = [perm[i*fold_size:(i+1)*fold_size].tolist() for i in range(5)]
    folds[-1].extend(perm[5*fold_size:].tolist())

    f1_per_fold_baseline = []
    f1_per_fold_corrected = []

    for fold_idx, val_idx in enumerate(folds):
        print(f"\n=== fold {fold_idx} ===")
        val_set = [all_data[i] for i in val_idx]
        train_set = [all_data[i] for i in range(len(all_data)) if i not in val_idx]
        # Concat all train (clip, feats, target, mask) into batches
        all_feats = np.concatenate([d["feats"][d["mask"]] for d in train_set if d["mask"].any()], axis=0)
        all_target = np.concatenate([d["target"][d["mask"]] for d in train_set if d["mask"].any()], axis=0)
        print(f"  train pairs: {len(all_target)}")
        x = torch.from_numpy(all_feats).to("cuda")
        y = torch.from_numpy(all_target).to("cuda")

        model = OffsetMLP(in_dim=all_feats.shape[1], hidden=hidden).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        for epoch in range(n_epochs):
            model.train()
            perm_e = torch.randperm(len(x))
            losses = []
            batch = 256
            for i in range(0, len(x), batch):
                idx = perm_e[i:i+batch]
                opt.zero_grad()
                pred = model(x[idx])
                loss = loss_fn(pred, y[idx])
                loss.backward(); opt.step()
                losses.append(float(loss.item()))
            if (epoch + 1) % 20 == 0:
                wandb.log({"fold": fold_idx, "epoch": epoch,
                            "train_loss": sum(losses)/len(losses)})

        # Eval on val
        model.eval()
        f1_base, f1_corr = [], []
        with torch.no_grad():
            for d in val_set:
                if len(d["pred_iv"]) == 0:
                    f1_base.append(0.0); f1_corr.append(0.0); continue
                # Baseline
                f1_base.append(score_offset_f1(d["pred_iv"], d["pred_pi"],
                                                  d["gt_iv"], d["gt_pi"], ratio=0.2))
                # Apply correction to all predicted notes
                feats_t = torch.from_numpy(d["feats"]).to("cuda")
                deltas = model(feats_t).cpu().numpy()
                # Cap deltas to ±0.5s to avoid runaway predictions
                deltas = np.clip(deltas, -0.5, 0.5)
                corrected_iv = d["pred_iv"].copy()
                corrected_iv[:, 1] = corrected_iv[:, 1] + deltas
                # Ensure offset > onset
                corrected_iv[:, 1] = np.maximum(corrected_iv[:, 1], corrected_iv[:, 0] + 0.05)
                f1_corr.append(score_offset_f1(corrected_iv, d["pred_pi"],
                                                  d["gt_iv"], d["gt_pi"], ratio=0.2))
        mean_b = float(np.mean(f1_base))
        mean_c = float(np.mean(f1_corr))
        print(f"  fold {fold_idx} offset20 F1: baseline={mean_b:.4f}  corrected={mean_c:.4f}  delta={mean_c-mean_b:+.4f}")
        f1_per_fold_baseline.append(mean_b)
        f1_per_fold_corrected.append(mean_c)
        wandb.log({"fold": fold_idx, "f1_base": mean_b, "f1_corr": mean_c})

    bm, cm = float(np.mean(f1_per_fold_baseline)), float(np.mean(f1_per_fold_corrected))
    print(f"\n5-fold CV offset20 F1:")
    print(f"  baseline (heuristic):       {bm:.4f}")
    print(f"  corrected (B85 MLP):        {cm:.4f}")
    print(f"  delta:                      {cm-bm:+.4f}")
    print(f"  vs IAA ceiling 0.642:       gap={0.642-cm:+.4f}")
    wandb.summary.update({"f1_baseline": bm, "f1_corrected": cm, "delta": cm-bm})
    OUT_JSON.write_text(json.dumps({"f1_per_fold_baseline": f1_per_fold_baseline,
                                     "f1_per_fold_corrected": f1_per_fold_corrected,
                                     "f1_baseline": bm, "f1_corrected": cm,
                                     "delta": cm-bm, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    main(**vars(ap.parse_args()))
