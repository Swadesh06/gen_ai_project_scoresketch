"""HumScribe v3.4 Streamlit demo (B+2 work item 3).

Tabs:
- Transcribe: upload/record audio -> transcribe -> SVG + MIDI download
- Arrange:    melody-conditioned MusicGen-Melody arrangement (Stage 7)

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import io
import os
import tempfile
from pathlib import Path

import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_arranger(model_size: str = "melody"):
    """Cache MusicGen weights across reruns; first call is slow."""
    from humscribe.arrange.musicgen import _load
    return _load(model_size=model_size)


def transcribe_tab() -> None:
    st.header("Transcribe")
    st.caption("Upload a humming or instrument WAV; HumScribe produces a notated score.")
    uploaded = st.file_uploader("Audio", type=["wav", "mp3", "flac", "m4a"], key="trans_upload")
    cols = st.columns(3)
    with cols[0]:
        kind = st.selectbox("Input kind", ["humming", "piano", "guitar", "instrument"])
    with cols[1]:
        mode = st.selectbox("Mode", ["soft", "medium", "hard"])
    with cols[2]:
        pitch_model = st.selectbox("Pitch model", ["pesto_crepevoicing", "pesto", "crepe"])
    if not uploaded:
        st.info("Upload a clip to transcribe.")
        return
    if not st.button("Transcribe", type="primary"):
        return
    from humscribe.config import PipelineConfig
    from humscribe.pipeline import transcribe
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tf:
        tf.write(uploaded.getvalue())
        wav_path = tf.name
    cfg = PipelineConfig(input_kind=kind, mode=mode, pitch_model=pitch_model)
    with st.spinner("Transcribing…"):
        res = transcribe(wav_path, cfg=cfg)
    st.success(f"{res.n_notes} notes, BPM {int(round(res.bpm))}")
    st.components.v1.html(f"<div style='background:#fff'>{res.svg}</div>", height=600, scrolling=True)
    st.download_button("Download MusicXML", res.musicxml, file_name="transcription.musicxml",
                       mime="application/vnd.recordare.musicxml+xml")
    # Save the audio to session for the Arrange tab
    st.session_state["last_audio_path"] = wav_path


def arrange_tab() -> None:
    st.header("Arrange (generative — MusicGen-Melody)")
    st.caption("Conditions on the audio you uploaded above as a melody, plus a style prompt.")
    audio_path = st.session_state.get("last_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        st.info("Upload and transcribe an audio clip first (Transcribe tab).")
        return
    from humscribe.arrange.musicgen import PROMPT_PRESETS
    cols = st.columns(3)
    with cols[0]:
        preset = st.selectbox("Style preset", list(PROMPT_PRESETS.keys()))
    with cols[1]:
        size = st.selectbox("Model", ["melody", "melody-large"], index=0,
                            help="melody = 1.5B (fast), melody-large = 3.3B (highest quality)")
    with cols[2]:
        duration = st.slider("Duration (s)", 8, 30, 15)
    custom = st.text_input("Custom prompt (overrides preset if set)", "")
    if not st.button("Arrange", type="secondary"):
        return
    prompt = custom.strip() or PROMPT_PRESETS[preset]
    with st.spinner("Generating arrangement (first call loads weights)…"):
        from humscribe.arrange.musicgen import arrange
        wav_bytes = arrange(audio_path, prompt, duration_s=duration, model_size=size)
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("Download arrangement WAV", wav_bytes, file_name="arrangement.wav",
                       mime="audio/wav")


def main() -> None:
    st.set_page_config(page_title="HumScribe v3.4", layout="wide")
    st.title("HumScribe v3.4")
    st.caption("Audio → score → optional generative arrangement.")
    tab_t, tab_a = st.tabs(["Transcribe", "Arrange"])
    with tab_t:
        transcribe_tab()
    with tab_a:
        arrange_tab()


if __name__ == "__main__":
    main()
