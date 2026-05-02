"""Train mel-BiLSTM onset detector on Vocadito A1+A2 with 5-fold CV.
Evaluates note F1 per fold; reports mean."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import mir_eval
import numpy as np
import torch
import wandb

from humscribe.audio_io import load_audio
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.train.onset_mel import (
    MelOnsetConfig, align_to_grid, make_labels, make_mel_features,
    make_pitch_features, predict_mask, segment_via_onsets, train_loop,
)


VOC = Path("~/datasets/vocadito").expanduser()


def load_vocadito_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def build_one(clip_id: str, annotator: str, mel_cfg: MelOnsetConfig
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    wav = VOC / "Audio" / f"{clip_id}.wav"
    nf = VOC / "Annotations" / "Notes" / f"{clip_id}_notes{annotator}.csv"
    if not wav.exists() or not nf.exists():
        return None
    audio, sr = load_audio(str(wav), target_sr=mel_cfg.sr)
    mel = make_mel_features(audio, sr, mel_cfg)
    n_mel_frames = mel.shape[0]
    times = np.arange(n_mel_frames, dtype=np.float64) * (mel_cfg.hop_ms / 1000.0)
    pt, ph, pv = track_pitch_pesto(audio, sr)
    _, midi_g, vc_g = align_to_grid(pt, ph, pv, times)
    pitch_feats = make_pitch_features(midi_g, vc_g)
    feats = np.concatenate([mel, pitch_feats], axis=-1).astype(np.float32)
    gt_iv, gt_p = load_vocadito_notes(nf)
    labels = make_labels(gt_iv[:, 0], times, hop=mel_cfg.hop_ms / 1000.0)
    return feats, labels, times, midi_g, vc_g, gt_iv, gt_p


def score(notes: list[NoteEvent], iv: np.ndarray, hz: np.ndarray) -> tuple[float, float, float]:
    if not notes:
        return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, hz, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return float(p), float(r), float(f)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(epochs: int, hidden: int, k_folds: int, threshold: float) -> None:
    cfg_w = {
        "exp": "B19_mel_bilstm_kfold",
        "epochs": epochs, "hidden": hidden, "k_folds": k_folds,
        "threshold": threshold, "git_sha": git_sha(),
    }
    run = wandb.init(
        project="humscribe-v3.2", name=f"exp_B19_mel_kfold{k_folds}_h{hidden}",
        config=cfg_w, tags=["B19", "vocadito", "mel", "bilstm", "kfold"],
        dir="logs/wandb",
    )
    mc = MelOnsetConfig(hidden=hidden)
    annotators = ["A1", "A2"]
    audio_dir = VOC / "Audio"
    clip_ids = sorted(p.stem for p in audio_dir.glob("*.wav"))
    print(f"clips: {len(clip_ids)}, annotators: {annotators}")
    print("preparing all examples (this is the slow part — extract mel + PESTO once) ...")
    examples = {}
    for cid in clip_ids:
        for ann in annotators:
            d = build_one(cid, ann, mc)
            if d is not None:
                examples[(cid, ann)] = d
    print(f"prepared {len(examples)} (clip, annotator) pairs")

    rng = np.random.default_rng(0)
    all_clips = list(clip_ids)
    rng.shuffle(all_clips)
    folds = [all_clips[i::k_folds] for i in range(k_folds)]

    fold_f1s = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold_i in range(k_folds):
        val_clips = set(folds[fold_i])
        train_pairs = [(c, a) for (c, a) in examples if c not in val_clips]
        val_pairs = [(c, a) for (c, a) in examples if c in val_clips]
        print(f"\n=== fold {fold_i+1}/{k_folds}: train={len(train_pairs)}  val={len(val_pairs)} ===")
        tr_f = [examples[k][0] for k in train_pairs]
        tr_l = [examples[k][1] for k in train_pairs]
        va_f = [examples[k][0] for k in val_pairs]
        va_l = [examples[k][1] for k in val_pairs]
        model, history = train_loop(
            tr_f, tr_l, mc, va_f, va_l, epochs=epochs, batch_size=4, lr=1e-3, device=device,
        )
        for h in history:
            wandb.log({f"fold{fold_i+1}/{k}": v for k, v in h.items()})
        # downstream eval per val pair
        per_pair_f1 = []
        for (cid, ann) in val_pairs:
            feats, labels, times, midi_g, vc_g, gt_iv, gt_p = examples[(cid, ann)]
            mask = predict_mask(model, feats, threshold=threshold, device=device)
            from humscribe.notes import hz_to_midi
            hz_pred = np.array([midi_to_hz(midi_g[i]) if midi_g[i] > 0 else 0.0 for i in range(len(midi_g))])
            notes = segment_via_onsets(times, hz_pred, vc_g, mask)
            p, r, f = score(notes, gt_iv, gt_p)
            per_pair_f1.append(f)
        fold_mean = float(np.mean(per_pair_f1)) if per_pair_f1 else 0.0
        fold_f1s.append(fold_mean)
        print(f"  fold {fold_i+1} mean F1: {fold_mean:.3f}  (n_val_pairs={len(per_pair_f1)})")
        wandb.log({f"fold{fold_i+1}/mean_val_f1": fold_mean, "fold_idx": fold_i + 1})

    final_mean = float(np.mean(fold_f1s)) if fold_f1s else 0.0
    summary = {
        "mean_cv_f1": final_mean,
        "fold_f1s": fold_f1s,
        "k_folds": k_folds,
    }
    wandb.summary.update(summary)
    print(f"\n[B19] {k_folds}-fold mean F1: {final_mean:.3f}")
    out = Path(f"reports/_exp_B19_mel_bilstm_kfold{k_folds}.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nrun: {run.url}\njson: {out}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    main(args.epochs, args.hidden, args.k_folds, args.threshold)
