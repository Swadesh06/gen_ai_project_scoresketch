"""B86 — AMT continuation on a POLYPHONIC piano prompt (Phase E follow-up to B81).

B81 showed AMT generates 0 new events on a monophonic Vocadito prompt
(out of training distribution). B86 tests the same model on Bach BWV 854
(piano fugue transcription, 740+ notes, dense polyphony) — well within
the AMT training distribution (Lakh + GiantMIDI-Piano).

Pass criterion: AMT generates ≥ 50 new events in the requested 15s window.
"""
from __future__ import annotations
import json
import pickle
import subprocess
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM

from humscribe.notes import NoteEvent

CACHE = Path("/workspace/.cache/asap_yourmt3plus")
OUT_DIR = Path("outputs/amt_continuation")
OUT_JSON = Path("reports/_exp_B86_amt_polyphonic.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_b63_cache_notes(piece: str):
    safe = piece.replace("/", "__")
    cache = CACHE / f"{safe}.pkl"
    d = pickle.loads(cache.read_bytes())
    notes = []
    for x in d.get("notes", []):
        if isinstance(x, NoteEvent):
            notes.append(x); continue
        hz = x.get("hz"); mid = x.get("midi")
        if hz is None and mid is not None:
            hz = 440.0 * 2 ** ((mid - 69) / 12)
        if hz is None or mid is None: continue
        notes.append(NoteEvent(onset_s=x["on"], offset_s=x["off"],
                                pitch_midi=mid, pitch_hz=hz))
    return notes


def main(model_name: str = "stanford-crfm/music-medium-800k",
         piece: str = "Bach/Fugue/bwv_854",
         prompt_seconds: float = 30.0,
         continuation_seconds: float = 15.0,
         top_p: float = 0.95) -> None:
    cfg_w = {"git_sha": git_sha(), "model": model_name, "piece": piece,
             "prompt_seconds": prompt_seconds,
             "continuation_seconds": continuation_seconds, "top_p": top_p}
    run = wandb.init(project="humscribe-v3.2", name="exp_B86_amt_polyphonic",
                     config=cfg_w, tags=["B86", "amt", "polyphonic", "phase-e"],
                     dir="logs/wandb")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\nloading {piece} from B63 cache")
    notes = load_b63_cache_notes(piece)
    # Use first prompt_seconds of notes
    prompt_notes = [n for n in notes if n.onset_s <= prompt_seconds]
    print(f"  full notes: {len(notes)}; prompt notes (first {prompt_seconds}s): {len(prompt_notes)}")

    # Convert prompt notes to MIDI for AMT
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for n in prompt_notes:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=int(n.midi()),
                                             start=float(n.onset_s),
                                             end=float(n.offset_s)))
    pm.instruments.append(inst)
    midi_in = OUT_DIR / "bach_854_prompt.mid"
    pm.write(str(midi_in))

    from anticipation.convert import midi_to_events, events_to_midi
    from anticipation.sample import generate

    in_events = midi_to_events(str(midi_in))
    print(f"  input AMT events: {len(in_events)}")

    end_time = float(prompt_notes[-1].offset_s + continuation_seconds)
    t0 = time.time()
    with torch.no_grad():
        out_events = generate(model, start_time=prompt_notes[-1].offset_s,
                                end_time=end_time, inputs=in_events, top_p=top_p)
    gen_wall = time.time() - t0
    new_events = len(out_events) - len(in_events)
    print(f"  generated {len(out_events) - len(in_events)} new events in {gen_wall:.1f}s")

    midi_out = OUT_DIR / "bach_854_continuation.mid"
    try:
        events_to_midi(out_events, str(midi_out))
        pm_out = pretty_midi.PrettyMIDI(str(midi_out))
        new_notes = [n for inst in pm_out.instruments for n in inst.notes
                      if n.start > prompt_notes[-1].offset_s]
        print(f"  new notes in continuation MIDI: {len(new_notes)}")
    except Exception as e:
        print(f"  midi write/read failed: {e}")
        new_notes = []

    summary = {
        "n_input_notes": len(prompt_notes),
        "n_input_events": len(in_events),
        "n_output_events": len(out_events),
        "n_new_events": new_events,
        "n_new_notes": len(new_notes),
        "generation_wall_s": gen_wall,
        "continuation_seconds": continuation_seconds,
        "passes_50_new": len(new_notes) >= 50,
    }
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    wandb.summary.update(summary)
    OUT_JSON.write_text(json.dumps({"summary": summary, "config": cfg_w}, indent=2))
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="stanford-crfm/music-medium-800k")
    ap.add_argument("--piece", default="Bach/Fugue/bwv_854")
    ap.add_argument("--prompt-seconds", type=float, default=30.0)
    ap.add_argument("--continuation-seconds", type=float, default=15.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    main(**vars(ap.parse_args()))
