"""HumScribe v3.2 — humming/instrument audio to MusicXML/SVG."""
from humscribe.config import PipelineConfig, ModeConfig
from humscribe.notes import NoteEvent
from humscribe.pipeline import transcribe, TranscribeResult

__all__ = [
    "PipelineConfig",
    "ModeConfig",
    "NoteEvent",
    "transcribe",
    "TranscribeResult",
]

__version__ = "3.2.0"
