"""MusicXML + SVG rendering. SVG falls back to a piano-roll string if music21
cannot find an external renderer (LilyPond/MuseScore)."""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import io
import math
import xml.sax.saxutils as sx

import numpy as np
from music21 import stream, note as m21note, meter, tempo as m21tempo, duration as m21dur

from humscribe.notes import NoteEvent


def build_stream(
    notes: Sequence[NoteEvent],
    bpm: float,
    time_sig: str = "4/4",
    tatum_onsets: np.ndarray | None = None,
    tatum_offsets: np.ndarray | None = None,
    tatums_per_beat: int = 12,
) -> stream.Stream:
    s = stream.Stream()
    s.append(meter.TimeSignature(time_sig))
    s.append(m21tempo.MetronomeMark(number=float(bpm) if bpm > 0 else 120.0))
    n_notes = len(notes)
    use_tatum = tatum_onsets is not None and tatum_offsets is not None and len(tatum_onsets) == n_notes
    for i, ev in enumerate(notes):
        midi_pitch = ev.midi()
        if midi_pitch <= 0:
            n = m21note.Rest()
            ql = ev.duration_s * (bpm / 60.0) if bpm > 0 else ev.duration_s
        else:
            n = m21note.Note(midi=midi_pitch)
            n.volume.velocity = ev.velocity
            if use_tatum:
                ql = (tatum_offsets[i] - tatum_onsets[i]) / float(tatums_per_beat)
            else:
                ql = ev.duration_s * (bpm / 60.0) if bpm > 0 else 1.0
        ql = max(float(ql), 1.0 / tatums_per_beat)
        n.duration = m21dur.Duration(ql)
        s.append(n)
    return s


def write_musicxml(s: stream.Stream, out_path: str | Path | None = None) -> str:
    if out_path is None:
        buf = io.BytesIO()
        from music21.musicxml.m21ToXml import GeneralObjectExporter
        exporter = GeneralObjectExporter()
        xml_bytes = exporter.parse(s)
        return xml_bytes.decode("utf-8") if isinstance(xml_bytes, bytes) else str(xml_bytes)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    s.write("musicxml", fp=str(p))
    return p.read_text()


def render_svg(s: stream.Stream, notes: Sequence[NoteEvent], bpm: float) -> str:
    """Real notation via Verovio (preferred) > music21+LilyPond > piano-roll fallback."""
    try:
        return _verovio_svg(s)
    except Exception:
        pass
    try:
        from music21 import environment
        env = environment.Environment()
        ms = env.get("musicxmlPath") or env.get("musescoreDirectPNGPath")
        lp = env.get("lilypondPath")
        if ms or lp:
            tmp = s.write("musicxml.svg")
            return Path(tmp).read_text()
    except Exception:
        pass
    return _pianoroll_svg(notes, bpm)


def _verovio_svg(s: stream.Stream) -> str:
    """Render music21 Stream via Verovio's MusicXML loader. Returns first page SVG.
    Raises on any error so render_svg falls back."""
    import verovio
    musicxml = write_musicxml(s)
    tk = verovio.toolkit()
    tk.setOptions({
        "scale": 40, "pageHeight": 2970, "pageWidth": 2100,
        "adjustPageHeight": True, "adjustPageWidth": False,
        "footer": "none", "header": "none",
    })
    if not tk.loadData(musicxml):
        raise RuntimeError("verovio could not load musicxml")
    n_pages = tk.getPageCount()
    if n_pages < 1:
        raise RuntimeError("verovio rendered no pages")
    return tk.renderToSVG(1)


def _pianoroll_svg(notes: Sequence[NoteEvent], bpm: float) -> str:
    if not notes:
        return _empty_svg("(no notes detected)")
    end = max(n.offset_s for n in notes)
    px_per_s = 60.0
    width = max(int(math.ceil(end * px_per_s)) + 40, 200)
    midis = [n.midi() for n in notes if n.midi() > 0]
    lo = min(midis) - 2 if midis else 60
    hi = max(midis) + 2 if midis else 72
    rows = max(hi - lo, 1)
    px_per_row = 8
    height = rows * px_per_row + 40
    parts = [
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="8" y="14" font-family="monospace" font-size="11" fill="#444">'
        f'humscribe pianoroll  bpm={bpm:.1f}  notes={len(notes)}</text>',
    ]
    for n in notes:
        m = n.midi()
        if m <= 0:
            continue
        x = 20 + n.onset_s * px_per_s
        w = max((n.offset_s - n.onset_s) * px_per_s, 1.5)
        y = 20 + (hi - m) * px_per_row
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{px_per_row - 1}" '
            f'fill="#3a6ea5" stroke="#1f3d5a" stroke-width="0.5"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _empty_svg(msg: str) -> str:
    safe = sx.escape(msg)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="80" viewBox="0 0 320 80">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f'<text x="12" y="44" font-family="monospace" font-size="14" fill="#888">{safe}</text>\n'
        "</svg>"
    )
