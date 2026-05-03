"""B50: BiLSTM with pitch-shift augmentation. 5x training data via librosa.effects.pitch_shift.
Test if more data unlocks the BiLSTM (B42b at 0.619 was data-bound)."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import librosa, mir_eval, numpy as np, torch, wandb

from humscribe.audio_io import load_audio
from humscribe.notes import NoteEvent, midi_to_hz, hz_to_midi
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.train.onset_mel import (
    MelOnsetConfig, MelOnsetBiLSTM, make_mel_features, make_labels,
    predict_mask, segment_via_onsets, _pad_batch, _val_metrics, _pos_weight,
)


VOC = Path("~/datasets/vocadito").expanduser()


def load_vocadito_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def build_one_aug(clip_id: str, annotator: str, mel_cfg: MelOnsetConfig, n_steps: int):
    wav = VOC / "Audio" / f"{clip_id}.wav"
    nf = VOC / "Annotations" / "Notes" / f"{clip_id}_notes{annotator}.csv"
    if not wav.exists() or not nf.exists():
        return None
    audio, sr = load_audio(str(wav), target_sr=mel_cfg.sr)
    if n_steps != 0:
        audio = librosa.effects.pitch_shift(y=audio.astype(np.float32), sr=sr, n_steps=n_steps)
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


def main(epochs: int, hidden: int, k_folds: int, threshold: float, augment_steps: int):
    aug_steps = list(range(-augment_steps, augment_steps + 1))
    cfg_w = {"exp": "B50_bilstm_aug", "epochs": epochs, "hidden": hidden,
             "k_folds": k_folds, "threshold": threshold,
             "aug_steps": aug_steps, "git_sha": git_sha()}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B50_bilstm_aug{augment_steps}_h{hidden}",
                      config=cfg_w, tags=["B50", "vocadito", "bilstm", "augmented"], dir="logs/wandb")
    mc = MelOnsetConfig(hidden=hidden)
    annotators = ["A1", "A2"]
    audio_dir = VOC / "Audio"
    clip_ids = sorted(p.stem for p in audio_dir.glob("*.wav"))
    print(f"clips: {len(clip_ids)}, augmentation steps: {aug_steps}")
    print("extracting features (this is the slow part) ...")
    examples = {}  # (cid, ann, steps) -> data
    for cid in clip_ids:
        for ann in annotators:
            for steps in aug_steps:
                d = build_one_aug(cid, ann, mc, steps)
                if d is not None:
                    examples[(cid, ann, steps)] = d
    print(f"prepared {len(examples)} (clip, annotator, steps) examples")
    rng = np.random.default_rng(0)
    rng.shuffle(clip_ids)
    folds = [clip_ids[i::k_folds] for i in range(k_folds)]
    fold_f1s = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for fi in range(k_folds):
        val = set(folds[fi])
        # Train on all augmented versions of train clips; val only on original (steps=0)
        tr_pairs = [(c, a, s) for (c, a, s) in examples if c not in val]
        va_pairs = [(c, a, s) for (c, a, s) in examples if c in val and s == 0]
        tr_f = [examples[k][0] for k in tr_pairs]
        tr_l = [examples[k][1] for k in tr_pairs]
        va_f = [examples[k][0] for k in va_pairs]
        va_l = [examples[k][1] for k in va_pairs]
        in_dim = tr_f[0].shape[1]
        cfg2 = MelOnsetConfig(hidden=hidden, n_mels=in_dim - 5)
        model = MelOnsetBiLSTM(cfg2, in_extra=5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        from torch import nn
        pos_weight = _pos_weight(tr_l).to(device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        n = len(tr_f); rng_e = np.random.default_rng(0)
        for ep in range(epochs):
            order = rng_e.permutation(n); model.train(); train_loss = 0.0
            for i in range(0, n, 8):  # bigger batch since more data
                ids = order[i:i + 8]
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
            wandb.log({f"fold{fi+1}/epoch": ep, f"fold{fi+1}/train_loss": train_loss})
        # eval
        f1s = []
        for (cid, ann, _) in va_pairs:
            feats, labels, times, midi_p, pv, gt_iv, gt_p = examples[(cid, ann, 0)]
            mask = predict_mask(model, feats, threshold=threshold, device=device)
            hz_pred = np.array([midi_to_hz(midi_p[i]) if midi_p[i] > 0 else 0.0 for i in range(len(midi_p))])
            notes = segment_via_onsets(times, hz_pred, pv, mask)
            _, _, f = score(notes, gt_iv, gt_p)
            f1s.append(f)
        mf = float(np.mean(f1s)) if f1s else 0.0
        fold_f1s.append(mf)
        print(f"fold {fi+1}: F1={mf:.3f}  (n_train={len(tr_pairs)}, n_val={len(va_pairs)})")
        wandb.log({f"fold{fi+1}/mean_val_f1": mf})
    summary = {"mean_cv_f1": float(np.mean(fold_f1s)) if fold_f1s else 0.0,
               "fold_f1s": fold_f1s, "k_folds": k_folds,
               "augment_steps": augment_steps}
    wandb.summary.update(summary)
    print(f"\n[B50] {k_folds}-fold mean F1: {summary['mean_cv_f1']:.3f}")
    out = Path(f"reports/_exp_B50_bilstm_aug{augment_steps}_k{k_folds}.json")
    out.write_text(json.dumps(summary, indent=2))
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--augment-steps", type=int, default=2,
                    help="aug shifts: -N..N semitones (so 2 -> 5 versions)")
    main(**vars(ap.parse_args()))
