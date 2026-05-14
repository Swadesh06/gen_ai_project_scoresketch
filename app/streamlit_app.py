"""HumScribe v3.4 Streamlit demo.

Tabs:
- Transcribe: pre-recorded demo or uploaded audio -> notated score
              (mode-aware extra inputs: soft = audio only; medium adds a
              BPM hint; hard adds BPM + key + time signature).
- Arrange:    melody-conditioned MusicGen-Melody arrangement (Stage 7).

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import io
import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st


KEY_OPTIONS = [
    "Auto (detect)",
    "C major", "G major", "D major", "A major", "E major", "B major", "F# major",
    "F major", "Bb major", "Eb major", "Ab major", "Db major",
    "A minor", "E minor", "B minor", "F# minor", "C# minor",
    "D minor", "G minor", "C minor", "F minor", "Bb minor", "Eb minor",
]
TIME_SIG_OPTIONS = ["Auto (detect)", "4/4", "3/4", "6/8", "2/4", "12/8", "9/8", "5/4"]
KIND_OPTIONS = ["humming", "piano", "instrument", "guitar"]
MODE_OPTIONS = ["soft", "medium", "hard"]
PITCH_OPTIONS = ["pesto_crepevoicing", "pesto", "crepe"]


@st.cache_resource(show_spinner=False)
def _load_arranger(model_size: str = "melody"):
    """Cache MusicGen weights across reruns; first call is slow."""
    from humscribe.arrange.musicgen import _load
    return _load(model_size=model_size)


@st.cache_data(show_spinner=False)
def _load_demo_meta(demos_dir: str) -> list[dict]:
    p = Path(demos_dir) / "demos_meta.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _render_svg(svg: str) -> None:
    """Render a music21/Verovio SVG inline. The renderer emits a fixed-
    width SVG; we wrap it in a scrollable div sized for typical scores."""
    # Force the SVG to fit the iframe width while keeping its aspect ratio.
    svg_responsive = re.sub(
        r'<svg([^>]*?)>',
        r'<svg\1 style="width:100%;height:auto;max-width:100%;display:block;">',
        svg,
        count=1,
    )
    html = f"""
    <div style="background:#fff;padding:14px;border:1px solid #ddd;
                border-radius:6px;overflow-x:auto;overflow-y:auto;">
      {svg_responsive}
    </div>
    """
    st.components.v1.html(html, height=1000, scrolling=True)


def _mode_inputs(mode: str, defaults: dict | None = None) -> dict:
    """Return {target_bpm, key_override, time_sig_override} based on mode.

    soft  -> all None (audio-only auto-detect)
    medium -> target_bpm provided
    hard  -> target_bpm + key + time_sig
    """
    out: dict[str, object] = {"target_bpm": None, "key_override": None, "time_sig_override": None}
    if mode == "soft":
        st.caption(":blue[**soft**] - audio only. Pipeline auto-detects tempo, key, and time signature.")
        return out
    bpm_default = int(defaults.get("bpm_hint", 110)) if defaults else 110
    if mode == "medium":
        st.caption(":blue[**medium**] - audio + tempo hint. Locks the beat tracker octave.")
        cols = st.columns(2)
        with cols[0]:
            bpm = st.number_input("Tempo hint (BPM)", min_value=30, max_value=240,
                                   value=bpm_default, step=1, key="m_bpm",
                                   help="Used by beat_this as the target_bpm; tells the beat tracker which octave is correct.")
        out["target_bpm"] = float(bpm)
    else:  # hard
        st.caption(":blue[**hard**] - audio + tempo + key + time signature. Maximum supervision.")
        cols = st.columns(3)
        with cols[0]:
            bpm = st.number_input("Tempo (BPM)", min_value=30, max_value=240,
                                   value=bpm_default, step=1, key="h_bpm")
            out["target_bpm"] = float(bpm)
        key_default = (defaults or {}).get("key") or "Auto (detect)"
        ts_default = (defaults or {}).get("time_sig") or "Auto (detect)"
        # If demo provided a key, pre-select it; otherwise default to "Auto"
        key_idx = KEY_OPTIONS.index(key_default) if key_default in KEY_OPTIONS else 0
        ts_idx = TIME_SIG_OPTIONS.index(ts_default) if ts_default in TIME_SIG_OPTIONS else 0
        with cols[1]:
            chosen_key = st.selectbox("Key", KEY_OPTIONS, index=key_idx, key="h_key")
            if chosen_key != "Auto (detect)":
                out["key_override"] = chosen_key
        with cols[2]:
            chosen_ts = st.selectbox("Time signature", TIME_SIG_OPTIONS, index=ts_idx, key="h_ts")
            if chosen_ts != "Auto (detect)":
                out["time_sig_override"] = chosen_ts
    return out


def transcribe_tab() -> None:
    st.header("Transcribe")
    st.caption("Pick a pre-recorded demo or upload audio; HumScribe produces a notated score.")

    # ------------- 1. Source: demo or upload --------------------------------
    demos_dir = Path(__file__).parent / "demos"
    demo_meta = _load_demo_meta(str(demos_dir))
    demo_files = sorted(demos_dir.glob("demo_*.wav")) if demos_dir.exists() else []

    selected_demo_path: Path | None = None
    selected_demo_meta: dict | None = None
    if demo_meta:
        st.markdown("**Quick demo (no upload required)** -- 2 piano clips, 3 humming clips:")
        label_map = {m["label"]: m for m in demo_meta}
        labels = ["(none - upload your own)"] + list(label_map.keys())
        choice = st.selectbox("Pre-recorded clip", labels, key="demo_select")
        if choice != labels[0]:
            selected_demo_meta = label_map[choice]
            selected_demo_path = demos_dir / selected_demo_meta["file"]
            st.audio(str(selected_demo_path), format="audio/wav")
            st.caption(f":grey[{selected_demo_meta['blurb']}]")
    elif demo_files:
        st.markdown("**Quick demo (no upload required)**")
        labels = ["(none - upload your own)"] + [d.name for d in demo_files]
        choice = st.selectbox("Pre-recorded clip", labels, key="demo_select")
        if choice != labels[0]:
            selected_demo_path = demos_dir / choice
            st.audio(str(selected_demo_path), format="audio/wav")
    uploaded = st.file_uploader("...or upload audio",
                                 type=["wav", "mp3", "flac", "m4a"],
                                 key="trans_upload")

    # ------------- 2. Defaults for kind / mode / pitch from demo metadata ---
    kind_default = (selected_demo_meta or {}).get("kind", "humming")
    mode_default = (selected_demo_meta or {}).get("mode_default", "soft")
    pitch_default = (selected_demo_meta or {}).get("pitch_model", "pesto_crepevoicing")

    cols = st.columns(3)
    with cols[0]:
        kind = st.selectbox("Input kind", KIND_OPTIONS,
                             index=KIND_OPTIONS.index(kind_default) if kind_default in KIND_OPTIONS else 0,
                             help="Demo auto-sets this; change if you upload your own.")
    with cols[1]:
        mode = st.selectbox("Mode", MODE_OPTIONS,
                             index=MODE_OPTIONS.index(mode_default) if mode_default in MODE_OPTIONS else 0,
                             help="soft = audio only; medium = + tempo hint; hard = + tempo + key + time signature.")
    with cols[2]:
        pitch_model = st.selectbox("Pitch model", PITCH_OPTIONS,
                                     index=PITCH_OPTIONS.index(pitch_default) if pitch_default in PITCH_OPTIONS else 0,
                                     disabled=(kind != "humming"),
                                     help="Pitch model only matters for the humming branch.")

    # ------------- 3. Mode-aware extra fields -------------------------------
    extras = _mode_inputs(mode, selected_demo_meta)

    # ------------- 4. Optional multi-take consensus -------------------------
    multi_take = st.checkbox("Multi-take consensus (2-3 takes)",
                              value=False, key="multi_take",
                              help="Upload 2-3 audio clips of the SAME melody. We keep notes that appear in >=2 of them within +-50 ms.")
    extra_uploads = []
    if multi_take:
        extra_uploads = st.file_uploader(
            "Additional takes (upload 2-3 of the same melody)",
            type=["wav", "mp3", "flac", "m4a"],
            accept_multiple_files=True, key="trans_upload_extra",
        ) or []

    if not uploaded and not selected_demo_path:
        st.info("Pick a pre-recorded demo or upload a clip to transcribe.")
        return
    if not st.button("Transcribe", type="primary"):
        return

    # ------------- 5. Build config, transcribe ------------------------------
    from humscribe.config import PipelineConfig
    from humscribe.pipeline import transcribe
    if selected_demo_path is not None:
        wav_path = str(selected_demo_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tf:
            tf.write(uploaded.getvalue())
            wav_path = tf.name
    cfg = PipelineConfig(
        input_kind=kind, mode=mode, pitch_model=pitch_model,
        target_bpm=extras["target_bpm"],
        key_override=extras["key_override"],
        time_sig_override=extras["time_sig_override"],
    )
    if multi_take and extra_uploads:
        from humscribe.eval.multi_take import consensus_transcribe
        paths = [wav_path]
        for u in extra_uploads:
            with tempfile.NamedTemporaryFile(suffix=Path(u.name).suffix, delete=False) as tf2:
                tf2.write(u.getvalue())
                paths.append(tf2.name)
        with st.spinner(f"Transcribing {len(paths)} takes and computing consensus..."):
            res = consensus_transcribe(paths, cfg=cfg)
        st.info(f"Multi-take: {len(paths)} takes consolidated -> {res.n_notes} consensus notes")
    else:
        with st.spinner("Transcribing..."):
            res = transcribe(wav_path, cfg=cfg)

    st.success(f"{res.n_notes} notes  -  BPM {int(round(res.bpm))}")

    # ------------- 6. Render the score SVG ----------------------------------
    st.subheader("Notated score")
    _render_svg(res.svg)

    # ------------- 7. Downloads + session handoff to Arrange tab ------------
    dcols = st.columns(3)
    with dcols[0]:
        st.download_button("Download MusicXML", res.musicxml,
                            file_name="transcription.musicxml",
                            mime="application/vnd.recordare.musicxml+xml")
    with dcols[1]:
        st.download_button("Download SVG", res.svg,
                            file_name="transcription.svg",
                            mime="image/svg+xml")
    with dcols[2]:
        midi_bytes = _notes_to_midi_bytes(res.notes, bpm=res.bpm)
        if midi_bytes is not None:
            st.download_button("Download MIDI", midi_bytes,
                                file_name="transcription.mid",
                                mime="audio/midi")
    st.session_state["last_audio_path"] = wav_path


def _notes_to_midi_bytes(notes, bpm: float) -> bytes | None:
    """Convert NoteEvent list to a MIDI file via pretty_midi.
    Returns None on failure (used as a graceful skip when MIDI dep missing)."""
    try:
        import pretty_midi
    except ImportError:
        return None
    pm = pretty_midi.PrettyMIDI(initial_tempo=max(float(bpm), 30.0))
    inst = pretty_midi.Instrument(program=0)
    for ev in notes:
        midi_pitch = ev.midi()
        if midi_pitch <= 0:
            continue
        inst.notes.append(pretty_midi.Note(
            velocity=int(getattr(ev, "velocity", 80)),
            pitch=int(midi_pitch),
            start=float(ev.onset_s),
            end=float(max(ev.offset_s, ev.onset_s + 0.05)),
        ))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def arrange_tab() -> None:
    st.header("Arrange (generative - MusicGen-Melody)")
    st.caption("Conditions on the audio you transcribed above as a melody, plus a style prompt.")
    audio_path = st.session_state.get("last_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        st.info("Upload and transcribe an audio clip first (Transcribe tab).")
        return
    from humscribe.arrange.musicgen import PROMPT_PRESETS
    cols = st.columns(3)
    with cols[0]:
        preset = st.selectbox("Style preset", list(PROMPT_PRESETS.keys()))
    with cols[1]:
        size = st.selectbox("Model", ["melody-large", "melody"], index=0,
                            help="melody-large = 3.3B (peak ~6.25 GB, highest quality, default); melody = 1.5B (peak ~4.31 GB, fast)")
    with cols[2]:
        duration = st.slider("Duration (s)", 8, 30, 15)
    custom = st.text_input("Custom prompt (overrides preset if set)", "")
    adapter_dir = "checkpoints/musicgen_lora_b77"
    adapter_choices = ["(none - base model)"]
    if os.path.isdir(adapter_dir):
        for s in sorted(os.listdir(adapter_dir)):
            if (os.path.isdir(os.path.join(adapter_dir, s))
                    and os.path.isfile(os.path.join(adapter_dir, s, "adapter_model.safetensors"))):
                adapter_choices.append(s)
    adapter = st.selectbox("LoRA adapter (B77 fine-tune)", adapter_choices,
                            help="Pick a saved adapter checkpoint for personalised style.")
    adapter_path = (None if adapter.startswith("(none")
                      else os.path.join(adapter_dir, adapter))
    if not st.button("Arrange", type="secondary"):
        return
    prompt = custom.strip() or PROMPT_PRESETS[preset]
    with st.spinner("Generating arrangement (first call loads weights)..."):
        from humscribe.arrange.musicgen import arrange
        wav_bytes = arrange(audio_path, prompt, duration_s=duration,
                              model_size=size, lora_adapter=adapter_path)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("Download arrangement WAV", wav_bytes,
                        file_name="arrangement.wav", mime="audio/wav")


def main() -> None:
    st.set_page_config(page_title="HumScribe v3.4", layout="wide")
    st.title("HumScribe v3.4")
    st.caption("Audio -> score -> optional generative arrangement.")
    tab_t, tab_a = st.tabs(["Transcribe", "Arrange"])
    with tab_t:
        transcribe_tab()
    with tab_a:
        arrange_tab()


if __name__ == "__main__":
    main()
