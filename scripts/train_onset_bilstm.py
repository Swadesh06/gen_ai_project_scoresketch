"""Train onset detector on Vocadito; evaluate with the resulting segmenter
on the held-out clips for COnP F1."""
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
from humscribe.train.onset_bilstm import (
    OnsetModelConfig, make_features, make_labels, predict_onsets,
    segment_via_onsets, train_loop,
)


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def split_clips(seed: int, train_frac: float = 0.75) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    audio_dir = VOC / "Audio"
    all_ids = sorted(p.stem for p in audio_dir.glob("*.wav"))
    rng.shuffle(all_ids)
    cut = int(round(len(all_ids) * train_frac))
    return all_ids[:cut], all_ids[cut:]


def build_examples(clip_ids: list[str], annotator: str, hop: float = 0.01) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    gt: list[tuple[np.ndarray, np.ndarray]] = []
    times_list: list[np.ndarray] = []
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    for cid in clip_ids:
        wav = audio_dir / f"{cid}.wav"
        nf = notes_dir / f"{cid}_notes{annotator}.csv"
        if not wav.exists() or not nf.exists():
            continue
        audio, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_pesto(audio, sr)
        gt_iv, gt_p = load_notes(nf)
        f = make_features(hz, vc)
        l = make_labels(gt_iv[:, 0], t, hop=hop)
        feats.append(f); labels.append(l); gt.append((gt_iv, gt_p)); times_list.append(t)
    return feats, labels, gt, times_list


def score_clip(notes: list[NoteEvent], iv: np.ndarray, hz: np.ndarray) -> tuple[float, float, float]:
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


def main(annotator: str, seed: int, epochs: int, threshold: float, hidden: int) -> None:
    train_ids, val_ids = split_clips(seed)
    print(f"train: {len(train_ids)} clips, val: {len(val_ids)} clips")

    cfg = {
        "exp": "B10_onset_bilstm",
        "annotator": annotator,
        "seed": seed,
        "epochs": epochs,
        "threshold": threshold,
        "hidden": hidden,
        "git_sha": git_sha(),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
    }
    run = wandb.init(
        project="humscribe-v3.2",
        name=f"exp_B10_onset_bilstm_seed{seed}_h{hidden}",
        config=cfg,
        tags=["B10", "vocadito", "onset", "bilstm"],
        dir="logs/wandb",
    )

    print("extracting train features ...")
    tr_f, tr_l, _, _ = build_examples(train_ids, annotator)
    print("extracting val features ...")
    va_f, va_l, va_gt, va_t = build_examples(val_ids, annotator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = OnsetModelConfig(in_dim=3, hidden=hidden, layers=2, dropout=0.2)
    model, history = train_loop(
        tr_f, tr_l, cfg=model_cfg, epochs=epochs, batch_size=4, lr=1e-3,
        device=device, val_features=va_f, val_labels=va_l,
    )
    for h in history:
        wandb.log(h)

    f1s, ps, rs = [], [], []
    for i, cid in enumerate(val_ids):
        if i >= len(va_f):
            break
        hz = None; vc = None  # rebuild from features (col 0 is midi_norm; we also need raw hz)
        wav = VOC / "Audio" / f"{cid}.wav"
        nf = VOC / "Annotations" / "Notes" / f"{cid}_notes{annotator}.csv"
        if not wav.exists() or not nf.exists():
            continue
        audio, sr = load_audio(str(wav), target_sr=22050)
        t, hz, vc = track_pitch_pesto(audio, sr)
        onset_mask = predict_onsets(model, hz, vc, threshold=threshold, device=device)
        notes = segment_via_onsets(t, hz, vc, onset_mask)
        gt_iv, gt_p = load_notes(nf)
        p, r, f = score_clip(notes, gt_iv, gt_p)
        f1s.append(f); ps.append(p); rs.append(r)
        wandb.log({f"val/{cid}/F1": f, f"val/{cid}/P": p, f"val/{cid}/R": r})
        print(f"{cid:20s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  (notes={len(notes)})")

    summary = {
        "mean_val_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_val_p": float(np.mean(ps)) if ps else 0.0,
        "mean_val_r": float(np.mean(rs)) if rs else 0.0,
        "n_val": len(f1s),
        "history_last": history[-1] if history else {},
    }
    wandb.summary.update(summary)
    print(f"\n[B10] mean val F1: {summary['mean_val_f1']:.3f} (over {summary['n_val']} held-out clips)")
    out = Path(f"reports/_exp_B10_onset_bilstm_seed{seed}_h{hidden}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "history": history,
                                "train_ids": train_ids, "val_ids": val_ids}, indent=2))

    ckpt_dir = Path(f"checkpoints/onset_bilstm_seed{seed}_h{hidden}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model_cfg.__dict__,
                "summary": summary}, ckpt_dir / "best.pt")
    print(f"checkpoint: {ckpt_dir/'best.pt'}")
    print(f"run: {run.url}")
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    main(args.annotator, args.seed, args.epochs, args.threshold, args.hidden)
