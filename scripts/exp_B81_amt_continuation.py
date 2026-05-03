"""B81 — Anticipatory Music Transformer continuation demo (Phase D).

Bolts on a "score continuation" feature using stanford-crfm/music-medium-800k
(Thickstun et al. 2024). Takes a transcribed humming MIDI and generates a
continuation. The combined sequence is exported as MIDI + rendered as SVG.

This is a generative-AI demo flourish: a different family of generative
model (autoregressive transformer over anticipatory MIDI tokens) than
MusicGen (autoregressive transformer over EnCodec audio tokens).

Pass criterion: end-to-end MIDI -> AMT -> rendered SVG without crash.
Quality is qualitative.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM

from humscribe.config import PipelineConfig
from humscribe.pipeline import transcribe

OUT_DIR = Path("outputs/amt_continuation")
OUT_JSON = Path("reports/_exp_B81_amt_continuation.json")
MELODY_AUDIO = Path("/home/swadesh/datasets/vocadito/Audio/vocadito_1.wav")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(model_name: str = "stanford-crfm/music-medium-800k",
         continuation_seconds: float = 15.0, top_p: float = 0.95) -> None:
    cfg_w = {"git_sha": git_sha(), "model": model_name,
             "continuation_seconds": continuation_seconds, "top_p": top_p}
    run = wandb.init(project="humscribe-v3.2", name="exp_B81_amt_continuation",
                     config=cfg_w, tags=["B81", "amt", "phase-d", "demo"],
                     dir="logs/wandb")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_name}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cuda").eval()
    print(f"  loaded in {time.time()-t0:.1f}s; "
          f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 1. Transcribe the humming clip via our pipeline
    print(f"\ntranscribing {MELODY_AUDIO.name}")
    cfg = PipelineConfig(input_kind="humming", mode="soft",
                          pitch_model="pesto_crepevoicing",
                          musicxml_path=str(OUT_DIR / "vocadito_1_in.musicxml"))
    res = transcribe(str(MELODY_AUDIO), cfg)
    print(f"  {res.n_notes} notes, bpm={res.bpm:.1f}")

    # 2. Convert NoteEvent list -> MIDI for AMT input
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # acoustic grand
    for n in res.notes:
        inst.notes.append(pretty_midi.Note(velocity=80,
                                             pitch=int(n.midi()),
                                             start=float(n.onset_s),
                                             end=float(n.offset_s)))
    pm.instruments.append(inst)
    midi_in = OUT_DIR / "vocadito_1_in.mid"
    pm.write(str(midi_in))
    print(f"  wrote {midi_in}")

    # 3. Generate continuation via AMT
    print(f"\ngenerating {continuation_seconds}s continuation")
    from anticipation.convert import midi_to_events, events_to_midi
    from anticipation.sample import generate

    in_events = midi_to_events(str(midi_in))
    print(f"  input events: {len(in_events)} events for "
          f"{(in_events[-3] if len(in_events) >= 3 else 0) / 1000.0:.2f}s")
    end_time = float(res.notes[-1].offset_s + continuation_seconds) if res.notes else continuation_seconds
    t0 = time.time()
    with torch.no_grad():
        out_events = generate(model, start_time=res.notes[-1].offset_s if res.notes else 0,
                                end_time=end_time, inputs=in_events, top_p=top_p)
    gen_wall = time.time() - t0
    print(f"  generation wall: {gen_wall:.1f}s")
    print(f"  output events: {len(out_events)}")

    # 4. Convert AMT output to MIDI
    midi_out = OUT_DIR / "vocadito_1_continuation.mid"
    events_to_midi(out_events, str(midi_out))
    print(f"  wrote {midi_out}")

    summary = {
        "n_input_notes": res.n_notes,
        "n_input_events": len(in_events),
        "n_output_events": len(out_events),
        "generation_wall_s": gen_wall,
        "model_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "midi_in_path": str(midi_in),
        "midi_out_path": str(midi_out),
        "continuation_seconds": continuation_seconds,
    }
    wandb.summary.update(summary)
    OUT_JSON.write_text(json.dumps({"summary": summary, "config": cfg_w}, indent=2))
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    print(f"  run: {run.url}")
    run.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="stanford-crfm/music-medium-800k")
    ap.add_argument("--continuation-seconds", type=float, default=15.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    main(**vars(ap.parse_args()))
