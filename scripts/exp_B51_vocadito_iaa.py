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


def score(ref_iv, ref_p, est_iv, est_p):
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
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
        # A1 as ref, A2 as est
        p_ab, r_ab, f_ab = score(a_iv, a_p, b_iv, b_p)
        # And reverse to confirm symmetry
        p_ba, r_ba, f_ba = score(b_iv, b_p, a_iv, a_p)
        rows.append({"clip": cid, "n_a1": len(a_iv), "n_a2": len(b_iv),
                     "f_ab": f_ab, "f_ba": f_ba, "f_mean": (f_ab + f_ba) / 2})
        print(f"  {cid:25s}  A1->A2 F={f_ab:.3f}  A2->A1 F={f_ba:.3f}  (n={len(a_iv)},{len(b_iv)})")
    if not rows:
        print("no overlapping clips"); run.finish(); return
    f_means = [r["f_mean"] for r in rows]
    overall = float(np.mean(f_means))
    overall_sd = float(np.std(f_means))
    print(f"\nOverall IAA mean F1 = {overall:.3f} +/- {overall_sd:.3f}  (n_clips={len(rows)})")
    print(f"Min: {min(f_means):.3f}  Max: {max(f_means):.3f}")
    wandb.log({"iaa_mean_f1": overall, "iaa_std": overall_sd, "n_clips": len(rows)})
    out = Path("reports/_exp_B51_iaa.json")
    out.write_text(json.dumps({"per_clip": rows, "iaa_mean_f1": overall,
                               "iaa_std": overall_sd, "n_clips": len(rows)}, indent=2))
    print(f"\nWrote {out}")
    run.finish()


if __name__ == "__main__":
    main()
