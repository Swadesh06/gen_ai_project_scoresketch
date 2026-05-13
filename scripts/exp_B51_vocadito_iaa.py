"""B51: Vocadito A1 vs A2 inter-annotator agreement.
Compute the ceiling for any supervised note transcriber on Vocadito.
If A1<->A2 F1 ~0.7, then our 0.665 is at ceiling and we can't push higher
without aggregating annotators or changing the metric."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb


VOC = Path("~/datasets/vocadito").expanduser()


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def score(ref_iv, ref_p, est_iv, est_p, offset_ratio=None):
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=offset_ratio, offset_min_tolerance=0.05)
    return float(p), float(r), float(f)


def git_sha() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    notes_dir = VOC / "Annotations" / "Notes"
    a1_files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B51_vocadito_iaa",
                     config={"git_sha": git_sha()}, tags=["B51", "vocadito", "iaa"], dir="logs/wandb")
    rows = []
    for af in a1_files:
        cid = af.stem.replace("_notesA1", "")
        bf = notes_dir / f"{cid}_notesA2.csv"
        if not bf.exists(): continue
        a_iv, a_p = load_notes(af)
        b_iv, b_p = load_notes(bf)
        f_no = score(a_iv, a_p, b_iv, b_p, offset_ratio=None)[2]
        f_o20 = score(a_iv, a_p, b_iv, b_p, offset_ratio=0.2)[2]
        f_o50 = score(a_iv, a_p, b_iv, b_p, offset_ratio=0.5)[2]
        rows.append({"clip": cid, "n_a1": len(a_iv), "n_a2": len(b_iv),
                     "f_ab": f_no, "f_ba": f_no, "f_mean": f_no,
                     "f_o20": f_o20, "f_o50": f_o50})
        print(f"  {cid:25s}  no_off={f_no:.3f}  off20={f_o20:.3f}  off50={f_o50:.3f}")
    if not rows:
        print("no overlapping clips"); run.finish(); return
    means = {k: float(np.mean([r[k] for r in rows])) for k in ("f_mean", "f_o20", "f_o50")}
    print(f"\nIAA means (n={len(rows)} clips):")
    for k, v in means.items(): print(f"  {k:10s} = {v:.3f}")
    wandb.summary.update({"iaa_no_offset": means["f_mean"],
                           "iaa_offset20": means["f_o20"],
                           "iaa_offset50": means["f_o50"],
                           "n_clips": len(rows)})
    out = Path("reports/_exp_B51_iaa.json")
    out.write_text(json.dumps({"per_clip": rows, "means": means, "n_clips": len(rows)}, indent=2))
    print(f"\nWrote {out}")
    run.finish()


if __name__ == "__main__":
    main()
