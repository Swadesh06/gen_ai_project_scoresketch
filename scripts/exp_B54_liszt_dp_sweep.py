"""B54: Liszt-only DP sweep with TPB=24/48, pj∈{7,12,24}, extended durations.
Run on oracle inputs first (since B53 showed Liszt is DP-bound, not upstream).
Then re-run on actual ByteDance outputs to confirm a fix transfers."""
from __future__ import annotations
import json, subprocess, pickle
from pathlib import Path
import mir_eval, numpy as np, pretty_midi, wandb

from humscribe.notes import NoteEvent
from humscribe.rhythm.viterbi_quantize import viterbi_quantize_rhythm
from humscribe.rhythm.voice_tracking import VoiceTrackConfig, assign_voices, per_voice_durations

ASAP = Path("~/datasets/asap").expanduser()
CACHE_DIR = Path("/workspace/.cache/asap_bytedance")
ALLOWED_BEATS = np.array([0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

# TPB=24 default
DUR_TPB24 = np.array([3, 4, 6, 9, 8, 12, 18, 16, 24, 36, 32, 48, 72, 96], dtype=np.int64)
# TPB=48 extended: includes 64th=3, dot-32nd=9, 32nd-trip=4, dot-16th=36, etc.
DUR_TPB48 = np.array([3, 6, 8, 12, 18, 16, 24, 36, 32, 48, 72, 64, 96, 144, 192], dtype=np.int64)
# TPB=24 extended: add quintuplet sub-beats and 5/16, 7/16
DUR_TPB24_EXT = np.array([3, 4, 6, 9, 8, 12, 18, 16, 21, 24, 30, 36, 32, 42, 48, 60, 72, 96], dtype=np.int64)


def load_cached(piece_dir: Path):
    key = piece_dir.relative_to(ASAP).as_posix().replace("/", "__")
    cache_path = CACHE_DIR / f"{key}.pkl"
    with open(cache_path, "rb") as f:
        d = pickle.load(f)
    notes = [NoteEvent(onset_s=x["on"], offset_s=x["off"], pitch_midi=x["midi"], pitch_hz=x["hz"], velocity=x["vel"], confidence=x["conf"]) for x in d["notes"]]
    return notes, d["beats"]


def load_midi_notes(mid: Path):
    pm = pretty_midi.PrettyMIDI(str(mid))
    iv = []; pi = []; events = []
    for inst in pm.instruments:
        for n in inst.notes:
            hz = 440.0 * 2 ** ((n.pitch - 69) / 12)
            iv.append([n.start, n.end]); pi.append(hz)
            events.append(NoteEvent(onset_s=n.start, offset_s=n.end, pitch_midi=n.pitch, pitch_hz=hz, velocity=n.velocity, confidence=1.0))
    return np.array(iv), np.array(pi), sorted(events, key=lambda e: e.onset_s)


def snap_beats(d): return float(ALLOWED_BEATS[np.argmin(np.abs(ALLOWED_BEATS - d))])


def evaluate(notes, beats, gt_iv, gt_p, avg_beat, tpb, durations, pj):
    cfg = VoiceTrackConfig(pitch_jump=pj, time_gap_s=0.5)
    voices = assign_voices(notes, cfg)
    on_v, off_v = per_voice_durations(notes, voices)
    q_on, q_off = viterbi_quantize_rhythm(on_v, off_v, beats, tatums_per_beat=tpb, allowed_durations_tatums=durations)
    pred_durs = (q_off - q_on) / tpb
    gt_durs_q = (gt_iv[:, 1] - gt_iv[:, 0]) / avg_beat
    onsets = np.array([n.onset_s for n in notes]); offsets = np.array([n.offset_s for n in notes])
    est_iv = np.column_stack([onsets, np.maximum(offsets, onsets + 1e-3)])
    est_p = np.array([n.pitch_hz for n in notes])
    matched = mir_eval.transcription.match_notes(gt_iv, gt_p, est_iv, est_p, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if not matched: return 0.0
    gi = [m[0] for m in matched]; pi = [m[1] for m in matched]
    pd = pred_durs[pi]; gd = gt_durs_q[gi]
    pd_s = np.array([snap_beats(float(x)) for x in pd])
    gd_s = np.array([snap_beats(float(x)) for x in gd])
    return float(np.mean(pd_s == gd_s))


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"


def main():
    run = wandb.init(project="humscribe-v3.2", name="exp_B54_liszt_dp_sweep",
                     config={"git_sha": git_sha()}, tags=["B54", "asap", "liszt", "dp"], dir="logs/wandb")
    piece = ASAP / "Liszt/Sonata"
    pm = pretty_midi.PrettyMIDI(str(piece / "midi_score.mid"))
    gt_beats = pm.get_beats()
    gt_iv, gt_p, gt_events = load_midi_notes(piece / "midi_score.mid")
    avg_beat = float(np.diff(gt_beats).mean())
    print(f"Liszt: {len(gt_events)} notes, {len(gt_beats)} GT beats, avg_beat={avg_beat:.3f}s")
    # Oracle: GT notes + GT beats. Sweep TPB / durations / pj.
    print("\n== Oracle inputs ==")
    rows = []
    for tpb, durations, dlabel in [(24, DUR_TPB24, "TPB24-default"),
                                    (24, DUR_TPB24_EXT, "TPB24-ext"),
                                    (48, DUR_TPB48, "TPB48-ext")]:
        for pj in (7, 12, 24):
            sn = evaluate(gt_events, gt_beats, gt_iv, gt_p, avg_beat, tpb, durations, pj)
            rows.append({"input": "oracle", "tpb": tpb, "dur": dlabel, "pj": pj, "snap": sn})
            print(f"  oracle  tpb={tpb}  dur={dlabel:15s}  pj={pj:2d}  snap={sn:.3f}")
            wandb.log({"oracle/tpb": tpb, "oracle/pj": pj, "oracle/snap": sn})
    # Actual inputs: ByteDance + beat_this from cache
    cache_path = CACHE_DIR / "Liszt__Sonata.pkl"
    if cache_path.exists():
        notes, beats = load_cached(piece)
        avg_beat_a = float(np.diff(beats).mean())
        print(f"\n== Actual inputs (cached) ==  ({len(notes)} ByteDance notes, {len(beats)} beats, avg={avg_beat_a:.3f})")
        for tpb, durations, dlabel in [(24, DUR_TPB24, "TPB24-default"),
                                       (24, DUR_TPB24_EXT, "TPB24-ext"),
                                       (48, DUR_TPB48, "TPB48-ext")]:
            for pj in (7, 12, 24):
                sn = evaluate(notes, beats, gt_iv, gt_p, avg_beat_a, tpb, durations, pj)
                rows.append({"input": "actual", "tpb": tpb, "dur": dlabel, "pj": pj, "snap": sn})
                print(f"  actual  tpb={tpb}  dur={dlabel:15s}  pj={pj:2d}  snap={sn:.3f}")
                wandb.log({"actual/tpb": tpb, "actual/pj": pj, "actual/snap": sn})
    rows_o = [r for r in rows if r["input"] == "oracle"]
    rows_a = [r for r in rows if r["input"] == "actual"]
    print("\nTop 3 oracle:"); rows_o.sort(key=lambda r: -r["snap"])
    for r in rows_o[:3]: print(f"  snap={r['snap']:.3f}  tpb={r['tpb']}  dur={r['dur']}  pj={r['pj']}")
    if rows_a:
        print("\nTop 3 actual:"); rows_a.sort(key=lambda r: -r["snap"])
        for r in rows_a[:3]: print(f"  snap={r['snap']:.3f}  tpb={r['tpb']}  dur={r['dur']}  pj={r['pj']}")
    out = Path("reports/_exp_B54_liszt_dp_sweep.json")
    out.write_text(json.dumps({"rows": rows}, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
