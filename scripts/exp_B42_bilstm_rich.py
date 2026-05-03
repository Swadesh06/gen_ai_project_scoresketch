"""B42: re-do B19 mel-BiLSTM but with PESTO + CREPE features (5-dim added) +
larger hidden + 5-fold CV. Maybe richer features unlock the BiLSTM."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import mir_eval, numpy as np, torch, wandb

from humscribe.audio_io import load_audio
from humscribe.notes import NoteEvent, midi_to_hz, hz_to_midi
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.train.onset_mel import (
    MelOnsetConfig, MelOnsetBiLSTM, make_mel_features, make_labels,
    predict_mask, segment_via_onsets, train_loop,
)


VOC = Path("~/datasets/vocadito").expanduser()


def load_vocadito_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows]); pi = np.array([float(r[1]) for r in rows])
    du = np.array([float(r[2]) for r in rows])
    return np.stack([on, on + du], axis=1), pi


def build_one(clip_id: str, annotator: str, mel_cfg: MelOnsetConfig):
    wav = VOC / "Audio" / f"{clip_id}.wav"
    nf = VOC / "Annotations" / "Notes" / f"{clip_id}_notes{annotator}.csv"
    if not wav.exists() or not nf.exists():
        return None
    audio, sr = load_audio(str(wav), target_sr=mel_cfg.sr)
    mel = make_mel_features(audio, sr, mel_cfg)
    n_frames = mel.shape[0]
    times = np.arange(n_frames, dtype=np.float64) * (mel_cfg.hop_ms / 1000.0)
    pt, ph, pv = track_pitch_pesto(audio, sr)
    ct, ch, cv = track_pitch_crepe(audio, sr)
    midi_p = np.where(ph > 0, np.array([hz_to_midi(float(h)) for h in ph]), 0.0)
    midi_c = np.where(ch > 0, np.array([hz_to_midi(float(h)) for h in ch]), 0.0)
    midi_p_g = np.interp(times, pt, midi_p).astype(np.float32)
    pv_g = np.interp(times, pt, pv).astype(np.float32)
    midi_c_g = np.interp(times, ct, midi_c).astype(np.float32)
    cv_g = np.interp(times, ct, cv).astype(np.float32)
    midi_diff = (midi_p_g - midi_c_g).astype(np.float32)
    extra = np.stack([
        ((midi_p_g - 60) / 24).astype(np.float32),
        pv_g, cv_g, midi_diff,
        ((pv_g + cv_g) / 2).astype(np.float32),
    ], axis=-1)
    feats = np.concatenate([mel, extra], axis=-1).astype(np.float32)
    gt_iv, gt_p = load_vocadito_notes(nf)
    labels = make_labels(gt_iv[:, 0], times, hop=mel_cfg.hop_ms / 1000.0)
    return feats, labels, times, midi_p_g, pv_g, gt_iv, gt_p


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


def main(epochs: int, hidden: int, k_folds: int, threshold: float):
    cfg_w = {"exp": "B42_bilstm_rich", "epochs": epochs, "hidden": hidden,
             "k_folds": k_folds, "threshold": threshold, "git_sha": git_sha()}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B42_bilstm_rich_h{hidden}_k{k_folds}",
                      config=cfg_w, tags=["B42", "vocadito", "bilstm", "rich"], dir="logs/wandb")
    mc = MelOnsetConfig(hidden=hidden)
    annotators = ["A1", "A2"]
    audio_dir = VOC / "Audio"
    clip_ids = sorted(p.stem for p in audio_dir.glob("*.wav"))
    print(f"clips: {len(clip_ids)}, extracting features ...")
    examples = {}
    for cid in clip_ids:
        for ann in annotators:
            d = build_one(cid, ann, mc)
            if d is not None:
                examples[(cid, ann)] = d
    print(f"prepared {len(examples)} (clip, annotator) pairs")
    rng = np.random.default_rng(0)
    rng.shuffle(clip_ids)
    folds = [clip_ids[i::k_folds] for i in range(k_folds)]
    fold_f1s = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for fi in range(k_folds):
        val = set(folds[fi])
        tr_pairs = [(c, a) for (c, a) in examples if c not in val]
        va_pairs = [(c, a) for (c, a) in examples if c in val]
        tr_f = [examples[k][0] for k in tr_pairs]
        tr_l = [examples[k][1] for k in tr_pairs]
        va_f = [examples[k][0] for k in va_pairs]
        va_l = [examples[k][1] for k in va_pairs]
        in_dim = tr_f[0].shape[1]
        # Build a fresh model with correct in_dim (mel + 5 extra features)
        cfg2 = MelOnsetConfig(hidden=hidden, n_mels=in_dim - 5)
        model = MelOnsetBiLSTM(cfg2, in_extra=5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        from humscribe.train.onset_mel import _pos_weight, _pad_batch, _val_metrics
        from torch import nn
        pos_weight = _pos_weight(tr_l).to(device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        n = len(tr_f); rng_e = np.random.default_rng(0)
        hist = []
        for ep in range(epochs):
            order = rng_e.permutation(n); model.train(); train_loss = 0.0
            for i in range(0, n, 4):
                ids = order[i:i + 4]
                xs = [torch.from_numpy(tr_f[j]) for j in ids]
                ys = [torch.from_numpy(tr_l[j]) for j in ids]
                xb, yb, mask = _pad_batch(xs, ys, device)
                logits = model(xb)
                loss = (bce(logits, yb) * mask).sum() / mask.sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += float(loss.detach()) * len(ids)
            train_loss /= n
            rec = {"epoch": ep, "train_loss": train_loss}
            vl, vp, vr, vf = _val_metrics(model, va_f, va_l, device)
            rec.update({"val_loss": vl, "val_p": vp, "val_r": vr, "val_f1": vf})
            hist.append(rec)
        for h in hist: wandb.log({f"fold{fi+1}/{k}": v for k, v in h.items()})
        f1s = []
        for (cid, ann) in va_pairs:
            feats, labels, times, midi_p, pv, gt_iv, gt_p = examples[(cid, ann)]
            mask = predict_mask(model, feats, threshold=threshold, device=device)
            hz_pred = np.array([midi_to_hz(midi_p[i]) if midi_p[i] > 0 else 0.0 for i in range(len(midi_p))])
            notes = segment_via_onsets(times, hz_pred, pv, mask)
            _, _, f = score(notes, gt_iv, gt_p)
            f1s.append(f)
        mf = float(np.mean(f1s)) if f1s else 0.0
        fold_f1s.append(mf)
        print(f"fold {fi+1}: F1={mf:.3f}")
        wandb.log({f"fold{fi+1}/mean_val_f1": mf})
    summary = {"mean_cv_f1": float(np.mean(fold_f1s)) if fold_f1s else 0.0,
               "fold_f1s": fold_f1s, "k_folds": k_folds}
    wandb.summary.update(summary)
    print(f"\n[B42] {k_folds}-fold mean F1: {summary['mean_cv_f1']:.3f}")
    out = Path(f"reports/_exp_B42_bilstm_rich_k{k_folds}_h{hidden}.json")
    out.write_text(json.dumps(summary, indent=2))
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    main(**vars(ap.parse_args()))
