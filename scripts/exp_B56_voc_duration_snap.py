"""B56: snap Vocadito note durations to musically-plausible subdivisions.
Estimate tempo from median inter-onset interval, then snap each note's duration
to {1/16, 1/12, 1/8, dot-1/8, 1/4, dot-1/4, 1/2, dot-1/2, 1, 2} of the beat.
Goal: push offset-20 F1 from 0.439 toward IAA ceiling 0.642 (+20pp room)."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import mir_eval, numpy as np, wandb

from humscribe.audio_io import load_audio
from humscribe.config import PipelineConfig
from humscribe.notes import NoteEvent, midi_to_hz
from humscribe.pipeline import transcribe


VOC = Path("~/datasets/vocadito").expanduser()
DUR_FRACS = np.array([1/16, 1/12, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4, 1.0, 1.5, 2.0])


def load_notes(p: Path):
    rows = [r.strip().split(",") for r in p.read_text().splitlines() if r.strip()]
    on = np.array([float(r[0]) for r in rows], dtype=np.float64)
    pi = np.array([float(r[1]) for r in rows], dtype=np.float64)
    du = np.array([float(r[2]) for r in rows], dtype=np.float64)
    return np.stack([on, on + du], axis=1), pi


def estimate_beat_s(notes: list[NoteEvent]) -> float:
    """Median IOI is roughly the eighth-note IOI for typical humming.
    Multiply by 2 to get the beat (quarter-note) IOI estimate."""
    if len(notes) < 4: return 0.5
    onsets = np.array([n.onset_s for n in notes])
    iois = np.diff(np.sort(onsets))
    iois = iois[(iois > 0.05) & (iois < 2.0)]
    if len(iois) == 0: return 0.5
    median_ioi = float(np.median(iois))
    # Heuristic: assume median IOI is the unit for note onsets (8th-ish in humming).
    # So beat = 2 * median_ioi
    return max(median_ioi * 2.0, 0.2)


def snap_durations(notes: list[NoteEvent], strategy: str) -> list[NoteEvent]:
    if not notes or strategy == "none":
        return notes
    beat_s = estimate_beat_s(notes)
    out = []
    for n in notes:
        d = max(n.offset_s - n.onset_s, 0.05)
        if strategy == "snap_only":
            d_beats = d / beat_s
            idx = int(np.argmin(np.abs(DUR_FRACS - d_beats)))
            new_d = float(DUR_FRACS[idx]) * beat_s
        elif strategy == "snap_or_extend":
            d_beats = d / beat_s
            idx = int(np.argmin(np.abs(DUR_FRACS - d_beats)))
            new_d = float(DUR_FRACS[idx]) * beat_s
            new_d = max(new_d, beat_s / 8)  # min: 1/16 of beat
        elif strategy == "extend_short":
            d_beats = d / beat_s
            new_d = d
            if d_beats < 1/12:
                idx = int(np.argmin(np.abs(DUR_FRACS - d_beats)))
                new_d = float(DUR_FRACS[idx]) * beat_s
        else:
            raise ValueError(strategy)
        new_off = n.onset_s + new_d
        # Ensure no overlap with next onset (clip if needed)
        out.append(NoteEvent(onset_s=n.onset_s, offset_s=new_off,
                             pitch_midi=n.pitch_midi, pitch_hz=n.pitch_hz,
                             velocity=n.velocity, confidence=n.confidence))
    # Clip overlap: if note i+1 onset < note i offset, shorten note i
    final = []
    for i, n in enumerate(out):
        end = n.offset_s
        if i + 1 < len(out):
            end = min(end, out[i + 1].onset_s - 0.005)
            end = max(end, n.onset_s + 0.05)
        final.append(NoteEvent(onset_s=n.onset_s, offset_s=end,
                                pitch_midi=n.pitch_midi, pitch_hz=n.pitch_hz,
                                velocity=n.velocity, confidence=n.confidence))
    return final


def transcribe_clip(wav: Path):
    cfg = PipelineConfig(mode="soft", input_kind="humming", pitch_model="pesto_crepevoicing")
    return transcribe(str(wav), cfg=cfg).notes


def score(notes, gt_iv, gt_p, offset_ratio):
    if not notes: return 0.0
    eiv = np.array([[n.onset_s, max(n.offset_s, n.onset_s + 1e-3)] for n in notes])
    ep = np.array([n.pitch_hz if n.pitch_hz else midi_to_hz(n.midi()) for n in notes])
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, eiv, ep, onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=offset_ratio, offset_min_tolerance=0.05)
    return float(f)


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    audio_dir = VOC / "Audio"
    notes_dir = VOC / "Annotations" / "Notes"
    files = sorted(notes_dir.glob("*_notesA1.csv"))
    run = wandb.init(project="humscribe-v3.2", name="exp_B56_voc_dursnap",
                     config={"git_sha": git_sha()}, tags=["B56", "vocadito", "duration"], dir="logs/wandb")
    rows = []
    strategies = ["none", "extend_short", "snap_or_extend", "snap_only"]
    for nf in files:
        cid = nf.stem.replace("_notesA1", "")
        wav = audio_dir / f"{cid}.wav"
        if not wav.exists(): continue
        gt_iv, gt_p = load_notes(nf)
        notes = transcribe_clip(wav)
        for s in strategies:
            snapped = snap_durations(notes, s)
            f0 = score(snapped, gt_iv, gt_p, None)
            f20 = score(snapped, gt_iv, gt_p, 0.2)
            f50 = score(snapped, gt_iv, gt_p, 0.5)
            rows.append({"clip": cid, "strategy": s, "f_no": f0, "f_20": f20, "f_50": f50})
        print(f"  {cid:25s}  none/o20={rows[-4]['f_20']:.3f}  ext/o20={rows[-3]['f_20']:.3f}  snap_or_ext/o20={rows[-2]['f_20']:.3f}  snap/o20={rows[-1]['f_20']:.3f}")
    if not rows:
        print("no data"); run.finish(); return
    means = {}
    for s in strategies:
        sub = [r for r in rows if r["strategy"] == s]
        means[s] = {
            "no_offset": float(np.mean([r["f_no"] for r in sub])),
            "offset20": float(np.mean([r["f_20"] for r in sub])),
            "offset50": float(np.mean([r["f_50"] for r in sub])),
        }
    print("\nMeans by strategy:")
    print(f"  {'strategy':18s}  {'no_offset':>10s}  {'offset20':>10s}  {'offset50':>10s}")
    for s in strategies:
        m = means[s]
        print(f"  {s:18s}  {m['no_offset']:>10.3f}  {m['offset20']:>10.3f}  {m['offset50']:>10.3f}")
        wandb.summary.update({f"{s}/no_offset": m["no_offset"],
                               f"{s}/offset20": m["offset20"],
                               f"{s}/offset50": m["offset50"]})
    out = Path("reports/_exp_B56_voc_dursnap.json")
    out.write_text(json.dumps({"means": means, "rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
