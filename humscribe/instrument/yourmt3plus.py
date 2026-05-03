"""YourMT3+ T5-seq2seq music transcription backend (B+2 work item 2).

Wraps the pre-release inference code from the official HuggingFace Spaces app
(`https://huggingface.co/spaces/mimbres/YourMT3`). The Spaces clone lives at
`/workspace/yourmt3_hf` and contains:
  - amt/src/                  Python package (model, utils, config)
  - amt/logs/2024/<exp>/checkpoints/<file>.ckpt   LFS-pulled weights

Default checkpoint is YPTF.MoE+Multi (noPS): the same default chosen by the
Spaces app, with the broadest multi-instrument coverage and Apache-2.0
license. ~5 GB VRAM at fp16 inference.

Author note: YourMT3+ is published as a runnable demo, not a clean pip
package. We mimic the app.py invocation exactly to avoid drift.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import os
import sys
from typing import Iterable

import numpy as np
import torch

from humscribe.notes import NoteEvent

YMT3_HOME = Path(os.environ.get("YOURMT3_HOME", "/workspace/yourmt3_hf"))
YMT3_PROJECT = "2024"

CHECKPOINTS = {
    # name -> (exp_id_for_dir, args list excluding precision/exp_id)
    "moe_multi_nops": (
        "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops",
        "last.ckpt",
        ["-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26",
         "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4",
         "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
         "-ac", "spec", "-hop", "300", "-atc", "1"],
    ),
}


@contextmanager
def _ymt3_cwd():
    """YourMT3 config hardcodes save_dir='amt/logs'; cwd must be the Spaces root
    (parent of amt/). model_helper lives at the Spaces root; model/* package
    lives at amt/src/."""
    src = YMT3_HOME / "amt" / "src"
    spaces_root = YMT3_HOME
    for p in (str(src), str(spaces_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(str(spaces_root))
    try:
        yield
    finally:
        os.chdir(cwd)


_MODEL_CACHE: dict[str, object] = {}


def _load(checkpoint: str = "moe_multi_nops",
          precision: str = "16",
          device: str | None = None) -> object:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown YourMT3+ checkpoint: {checkpoint!r}")
    if checkpoint in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint]
    exp_id, ckpt_file, base_args = CHECKPOINTS[checkpoint]
    ckpt_arg = f"{exp_id}@{ckpt_file}"
    args = [ckpt_arg, "-p", YMT3_PROJECT, *base_args, "-pr", precision]
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with _ymt3_cwd():
        from model_helper import load_model_checkpoint
        model = load_model_checkpoint(args=args, device="cpu")
    model.to(target)
    _MODEL_CACHE[checkpoint] = model
    return model


def transcribe_yourmt3plus(audio_path: str,
                            checkpoint: str = "moe_multi_nops",
                            precision: str = "16") -> list[NoteEvent]:
    """Transcribe an audio file with YourMT3+; returns a list of NoteEvent."""
    model = _load(checkpoint=checkpoint, precision=precision)
    with _ymt3_cwd():
        import soundfile as sf
        import torchaudio.functional as F  # type: ignore
        from utils.audio import slice_padded_array
        from utils.note2event import mix_notes
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes
        # Use soundfile to avoid torchaudio's torchcodec dependency. Returns
        # (samples, channels) float64; convert to (1, T) float32 first.
        audio_np, sr = sf.read(str(audio_path), always_2d=True)
        audio = torch.from_numpy(audio_np.T.astype("float32"))  # (channels, T)
        audio = torch.mean(audio, dim=0).unsqueeze(0)
        audio = F.resample(audio, sr, model.audio_cfg["sample_rate"])
        audio_segments = slice_padded_array(
            audio, model.audio_cfg["input_frames"], model.audio_cfg["input_frames"],
        )
        device = next(model.parameters()).device
        audio_segments = torch.from_numpy(audio_segments.astype("float32")).to(device).unsqueeze(1)
        with torch.inference_mode():
            pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
        n_items = audio_segments.shape[0]
        start_secs_file = [
            model.audio_cfg["input_frames"] * i / model.audio_cfg["sample_rate"]
            for i in range(n_items)
        ]
        per_channel: list[list] = []
        n_err_cnt = Counter()
        num_channels = model.task_manager.num_decoding_channels
        for ch in range(num_channels):
            arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
            zipped, _events, ne_err_cnt = model.task_manager.detokenize_list_batches(
                arr_ch, start_secs_file, return_events=True,
            )
            notes_ch, n_err_ch = merge_zipped_note_events_and_ties_to_notes(zipped)
            per_channel.append(notes_ch)
            n_err_cnt += n_err_ch
        notes = mix_notes(per_channel)
    return _convert(notes)


def _convert(ymt3_notes: Iterable) -> list[NoteEvent]:
    """Convert YourMT3+ Note dataclass into our NoteEvent.

    YourMT3+ Note has: onset, offset, pitch (MIDI), velocity, program,
    is_drum. We treat pitched events only — drums get filtered.
    """
    out: list[NoteEvent] = []
    for n in ymt3_notes:
        if getattr(n, "is_drum", False):
            continue
        midi = int(getattr(n, "pitch"))
        if midi < 1 or midi > 127:
            continue
        on = float(getattr(n, "onset"))
        off = float(getattr(n, "offset"))
        if off <= on:
            off = on + 0.05
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        vel = int(getattr(n, "velocity", 80))
        out.append(NoteEvent(
            onset_s=on, offset_s=off,
            pitch_midi=midi, pitch_hz=hz, velocity=vel, confidence=1.0,
        ))
    out.sort(key=lambda e: e.onset_s)
    return out
