"""B52: BiLSTM with HuBERT features (no augmentation, 5-fold CV).
Replace mel + PESTO/CREPE features with HuBERT-base 768-dim @ 50Hz embeddings.
Goal: close the gap between 0.665 (current heuristic) and 0.740 (Vocadito IAA ceiling).
"""
from __future__ import annotations
import argparse, json, subprocess, gc
from pathlib import Path
import librosa, mir_eval, numpy as np, torch, wandb
from torch import nn

from humscribe.audio_io import load_audio
from humscribe.notes import NoteEvent, midi_to_hz, hz_to_midi
from humscribe.pitch.crepe_track import track_pitch_crepe
from humscribe.pitch.pesto_track import track_pitch_pesto
from humscribe.train.onset_mel import (
    make_labels, _pad_batch, _pos_weight,
)


VOC = Path("~/datasets/vocadito").expanduser()
HOP_S = 0.020  # HuBERT base = 50Hz output
HUBERT_SR = 16000


def load_vocadito_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def build_one(clip_id: str, annotator: str, hubert_model, device):
    wav = VOC / "Audio" / f"{clip_id}.wav"
    nf = VOC / "Annotations" / "Notes" / f"{clip_id}_notes{annotator}.csv"
    if not wav.exists() or not nf.exists():
        return None
    audio_22k, sr_22k = load_audio(str(wav), target_sr=22050)
    # HuBERT path (16k)
    audio_16k = librosa.resample(audio_22k, orig_sr=sr_22k, target_sr=HUBERT_SR)
    with torch.no_grad():
        x = torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0).to(device)
        h = hubert_model(x).last_hidden_state.squeeze(0).cpu().numpy()
    n_frames = h.shape[0]
    times = np.arange(n_frames, dtype=np.float64) * HOP_S
    # Pitch path on 22k for monophonic note pitch
    pt, ph, pv = track_pitch_pesto(audio_22k, sr_22k)
    ct, ch, cv = track_pitch_crepe(audio_22k, sr_22k)
    midi_p = np.where(ph > 0, np.array([hz_to_midi(float(h_)) for h_ in ph]), 0.0)
    midi_p_g = np.interp(times, pt, midi_p).astype(np.float32)
    pv_g = np.interp(times, pt, pv).astype(np.float32)
    cv_g = np.interp(times, ct, cv).astype(np.float32)
    # Concatenate HuBERT (768) + pitch_norm + voicing_p + voicing_c (3 extra)
    extra = np.stack([
        ((midi_p_g - 60) / 24).astype(np.float32),
        pv_g, cv_g,
    ], axis=-1)
    feats = np.concatenate([h, extra], axis=-1).astype(np.float32)
    gt_iv, gt_p = load_vocadito_notes(nf)
    labels = make_labels(gt_iv[:, 0], times, hop=HOP_S)
    return feats, labels, times, midi_p_g, pv_g, gt_iv, gt_p


def predict_mask(model, feats, threshold=0.5, suppress_frames=2, device="cuda"):
    model.eval()
    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    is_on = probs > threshold
    out = np.zeros_like(is_on, dtype=bool)
    for i in range(len(is_on)):
        if is_on[i] and (i == 0 or not out[max(0, i - suppress_frames):i].any()):
            out[i] = True
    return out


def segment(times, midi_p, voicing, mask, vt=0.30, mns=0.05):
    if len(times) == 0: return []
    starts = np.where(mask)[0]
    if len(starts) == 0: return []
    notes = []
    dt = times[1] - times[0] if len(times) > 1 else 0.02
    for k, s in enumerate(starts):
        e = starts[k + 1] - 1 if k + 1 < len(starts) else len(times) - 1
        if voicing[s:e + 1].mean() < vt * 0.5: continue
        valid_mask = midi_p[s:e + 1] > 0
        if not valid_mask.any(): continue
        midi_med = float(np.median(midi_p[s:e + 1][valid_mask]))
        midi_int = int(round(midi_med)) if midi_med > 0 else 0
        on_t = float(times[s]); off_t = float(times[e]) + dt
        if (off_t - on_t) < mns: continue
        notes.append(NoteEvent(onset_s=on_t, offset_s=off_t,
                               pitch_hz=midi_to_hz(midi_med), pitch_midi=midi_int,
                               confidence=float(voicing[s:e + 1].mean())))
    return notes


def score(notes, iv, hz):
    if not notes: return 0.0, 0.0, 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    eh = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv, hz, eiv, eh, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


class HubertHead(nn.Module):
    def __init__(self, in_dim, hidden=192, layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, bidirectional=True,
                            batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main(epochs, hidden, k_folds, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"exp": "B52_hubert_bilstm", "epochs": epochs, "hidden": hidden,
           "k_folds": k_folds, "threshold": threshold, "git_sha": git_sha()}
    run = wandb.init(project="humscribe-v3.2", name=f"exp_B52_hubert_h{hidden}",
                     config=cfg, tags=["B52", "vocadito", "hubert"], dir="logs/wandb")
    from transformers import HubertModel
    print("loading hubert ...")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    for p in hubert.parameters(): p.requires_grad = False
    annotators = ["A1", "A2"]
    audio_dir = VOC / "Audio"
    clip_ids = sorted(p.stem for p in audio_dir.glob("*.wav"))
    print(f"clips: {len(clip_ids)}")
    print("extracting hubert features ...")
    examples = {}
    for cid in clip_ids:
        for ann in annotators:
            d = build_one(cid, ann, hubert, device)
            if d is not None:
                examples[(cid, ann)] = d
    print(f"prepared {len(examples)} examples")
    # Free hubert from GPU after extraction
    del hubert; gc.collect(); torch.cuda.empty_cache()
    rng = np.random.default_rng(0)
    rng.shuffle(clip_ids)
    folds = [clip_ids[i::k_folds] for i in range(k_folds)]
    fold_f1s = []
    in_dim = next(iter(examples.values()))[0].shape[1]
    print(f"in_dim={in_dim}")
    for fi in range(k_folds):
        val = set(folds[fi])
        tr_pairs = [(c, a) for (c, a) in examples if c not in val]
        va_pairs = [(c, a) for (c, a) in examples if c in val]
        tr_f = [examples[k][0] for k in tr_pairs]
        tr_l = [examples[k][1] for k in tr_pairs]
        model = HubertHead(in_dim, hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_w = _pos_weight(tr_l).to(device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        n = len(tr_f); rng_e = np.random.default_rng(0)
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
            wandb.log({f"fold{fi+1}/epoch": ep, f"fold{fi+1}/train_loss": train_loss})
        f1s = []
        for (cid, ann) in va_pairs:
            feats, labels, times, midi_p, pv, gt_iv, gt_p = examples[(cid, ann)]
            mask = predict_mask(model, feats, threshold=threshold, device=device)
            notes = segment(times, midi_p, pv, mask)
            _, _, f = score(notes, gt_iv, gt_p)
            f1s.append(f)
        mf = float(np.mean(f1s)) if f1s else 0.0
        fold_f1s.append(mf)
        print(f"fold {fi+1}: F1={mf:.3f}  (n_train={len(tr_pairs)}, n_val={len(va_pairs)})")
        wandb.log({f"fold{fi+1}/mean_val_f1": mf})
    summary = {"mean_cv_f1": float(np.mean(fold_f1s)) if fold_f1s else 0.0,
               "fold_f1s": fold_f1s, "k_folds": k_folds}
    wandb.summary.update(summary)
    print(f"\n[B52] {k_folds}-fold mean F1: {summary['mean_cv_f1']:.3f}")
    out = Path(f"reports/_exp_B52_hubert_k{k_folds}.json")
    out.write_text(json.dumps(summary, indent=2))
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    main(**vars(ap.parse_args()))
