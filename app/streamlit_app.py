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
    # Phase G G-7: one-click pre-recorded demo hums. Source = Vocadito (40
    # Creative-Commons-licensed clips; we ship 5 here, renamed to generic
    # labels). Demos load directly without manual upload.
    demos_dir = Path(__file__).parent / "demos"
    demo_files = sorted(demos_dir.glob("demo_*.wav")) if demos_dir.exists() else []
    selected_demo = None
    if demo_files:
        st.markdown("**Quick demo (no upload required)**")
        labels = ["(none — upload your own)"] + [d.name for d in demo_files]
        choice = st.selectbox("Pre-recorded hum", labels, key="demo_select")
        if choice != labels[0]:
            selected_demo = demos_dir / choice
            st.audio(str(selected_demo), format="audio/wav")
    uploaded = st.file_uploader("Audio", type=["wav", "mp3", "flac", "m4a"], key="trans_upload")
    cols = st.columns(3)
    with cols[0]:
        kind = st.selectbox("Input kind", ["humming", "piano", "guitar", "instrument"])
    with cols[1]:
        mode = st.selectbox("Mode", ["soft", "medium", "hard"])
    with cols[2]:
        pitch_model = st.selectbox("Pitch model", ["pesto_crepevoicing", "pesto", "crepe"])
    # Phase G G-14: multi-take averaging mode. The user records 2-3 takes
    # of the same melody; we transcribe each and consensus-vote notes that
    # appear in >=2 of N takes within +-50 ms of one another.
    multi_take = st.checkbox("Multi-take consensus (3 takes)",
                             value=False, key="multi_take",
                             help="Upload 2-3 audio clips of the SAME melody. We keep notes that appear in >=2 of them within +-50 ms.")
    extra_uploads = []
    if multi_take:
        extra_uploads = st.file_uploader(
            "Additional takes (upload 2-3 of the same melody)",
            type=["wav", "mp3", "flac", "m4a"],
            accept_multiple_files=True, key="trans_upload_extra",
        ) or []
    if not uploaded and not selected_demo:
        st.info("Pick a pre-recorded demo or upload a clip to transcribe.")
        return
    if not st.button("Transcribe", type="primary"):
        return
    from humscribe.config import PipelineConfig
    from humscribe.pipeline import transcribe
    if selected_demo is not None:
        wav_path = str(selected_demo)
    else:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tf:
            tf.write(uploaded.getvalue())
            wav_path = tf.name
    cfg = PipelineConfig(input_kind=kind, mode=mode, pitch_model=pitch_model)
    if multi_take and extra_uploads:
        from humscribe.eval.multi_take import consensus_transcribe
        paths = [wav_path]
        for u in extra_uploads:
            with tempfile.NamedTemporaryFile(suffix=Path(u.name).suffix, delete=False) as tf2:
                tf2.write(u.getvalue())
                paths.append(tf2.name)
        with st.spinner(f"Transcribing {len(paths)} takes and computing consensus..."):
            res = consensus_transcribe(paths, cfg=cfg)
        st.info(f"Multi-take: {len(paths)} takes consolidated → {res.n_notes} consensus notes")
    else:
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
        size = st.selectbox("Model", ["melody-large", "melody"], index=0,
                            help="melody-large = 3.3B (peak ~6.25 GB, highest quality, default); melody = 1.5B (peak ~4.31 GB, fast)")
    with cols[2]:
        duration = st.slider("Duration (s)", 8, 30, 15)
    custom = st.text_input("Custom prompt (overrides preset if set)", "")
    # B77: optional LoRA adapter for fine-tuned style/speaker
    import os as _os
    adapter_dir = "checkpoints/musicgen_lora_b77"
    adapter_choices = ["(none — base model)"]
    if _os.path.isdir(adapter_dir):
        for s in sorted(_os.listdir(adapter_dir)):
            if (_os.path.isdir(_os.path.join(adapter_dir, s))
                    and _os.path.isfile(_os.path.join(adapter_dir, s, "adapter_model.safetensors"))):
                adapter_choices.append(s)
    adapter = st.selectbox("LoRA adapter (B77 fine-tune)", adapter_choices,
                            help="Pick a saved adapter checkpoint for personalised style.")
    adapter_path = (None if adapter.startswith("(none")
                      else _os.path.join(adapter_dir, adapter))
    if not st.button("Arrange", type="secondary"):
        return
    prompt = custom.strip() or PROMPT_PRESETS[preset]
    with st.spinner("Generating arrangement (first call loads weights)…"):
        from humscribe.arrange.musicgen import arrange
        wav_bytes = arrange(audio_path, prompt, duration_s=duration,
                              model_size=size, lora_adapter=adapter_path)
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
