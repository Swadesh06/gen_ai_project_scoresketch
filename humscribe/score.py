"""MusicXML + SVG rendering. SVG falls back to a piano-roll string if music21
cannot find an external renderer (LilyPond/MuseScore).

B+2 item 1 fixes (per results_v1_evalution.md):
- 1.1: round MetronomeMark BPM to integer.
- 1.3: distinct render_tpb (default 12) requantizes ql away from 24/48-let positions
  while the metric path still uses tatums_per_beat=24 internally.
- 1.4: Krumhansl-Schmuckler key insertion when key_signature is None.
"""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import io
import math
import xml.sax.saxutils as sx

import numpy as np
from music21 import stream, note as m21note, meter, tempo as m21tempo, duration as m21dur, key as m21key
from music21.analysis.discrete import KrumhanslSchmuckler

from humscribe.notes import NoteEvent


def build_stream(
    notes: Sequence[NoteEvent],
    bpm: float,
    time_sig: str = "4/4",
    tatum_onsets: np.ndarray | None = None,
    tatum_offsets: np.ndarray | None = None,
    tatums_per_beat: int = 12,
    render_tpb: int | None = None,
    estimate_key: bool = True,
    enharmonic_spelling: bool = False,
    key_override: str | None = None,
) -> stream.Stream:
    """Build a music21 Stream from notes.

    `tatums_per_beat` is the metric-path resolution used by upstream DP.
    `render_tpb` (default = `tatums_per_beat` when None) is the resolution
    the rendered durations are snapped to; pass 12 to keep the SVG free of
    24-lets/48-lets while preserving 32nd-note metric accuracy upstream.
    """
    rtpb = int(render_tpb) if render_tpb else int(tatums_per_beat)
    s = stream.Stream()
    s.append(meter.TimeSignature(time_sig))
    bpm_int = int(round(float(bpm))) if bpm > 0 else 120
    s.append(m21tempo.MetronomeMark(number=bpm_int))
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
                if rtpb != int(tatums_per_beat):
                    on_r = int(round(float(tatum_onsets[i]) * rtpb / float(tatums_per_beat)))
                    off_r = int(round(float(tatum_offsets[i]) * rtpb / float(tatums_per_beat)))
                    if off_r <= on_r: off_r = on_r + 1
                    ql = (off_r - on_r) / float(rtpb)
                else:
                    ql = (tatum_offsets[i] - tatum_onsets[i]) / float(tatums_per_beat)
            else:
                ql = ev.duration_s * (bpm / 60.0) if bpm > 0 else 1.0
        ql = max(float(ql), 1.0 / float(rtpb))
        n.duration = m21dur.Duration(ql)
        s.append(n)
    inferred_key = None
    if key_override:
        # UI-supplied key string like "C major", "G major", "A minor". The
        # music21 Key constructor wants ("C", "major") not "C major", and
        # signals mode via tonic case ("c" -> minor), so we parse manually.
        try:
            parts = key_override.strip().split()
            tonic = parts[0] if parts else "C"
            mode = parts[1].lower() if len(parts) > 1 else "major"
            if mode not in {"major", "minor"}:
                mode = "major"
            k = m21key.Key(tonic, mode)
            ks = m21key.KeySignature(k.sharps)
            s.insert(0, ks)
            inferred_key = k
        except Exception:
            pass
    elif estimate_key:
        try:
            k = KrumhanslSchmuckler().getSolution(s)
            if k is not None:
                ks = m21key.KeySignature(k.sharps)
                s.insert(0, ks)
                inferred_key = k
        except Exception:
            pass
    if enharmonic_spelling:
        try:
            from humscribe.ensemble.me9_line_of_fifths import spell_with_line_of_fifths
            spell_with_line_of_fifths(s, key=inferred_key)
        except Exception:
            pass
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
    """Real notation via Verovio. Tries strict options first, then a relaxed
    retry with `mei-basic`-like settings, then a music21 + LilyPond bridge
    if installed. Only falls back to the piano-roll SVG when every engraved-
    notation path fails AND there are no notes to engrave.

    The Streamlit UI explicitly does not want the piano-roll output, so any
    Verovio failure is logged to stderr and surfaced via a banner SVG that
    instructs the user instead of silently returning bars.
    """
    import sys
    if not notes:
        return _empty_svg("(no notes detected)")
    # Pass 1: strict options (the production page layout we've used since B+1).
    try:
        return _verovio_svg(s)
    except Exception as e1:
        print(f"[render_svg] verovio strict failed: {type(e1).__name__}: {e1}",
              file=sys.stderr)
    # Pass 2: relaxed Verovio options. Some music21 outputs (long ties,
    # extreme tuplets, empty measures) trip the strict page builder; the
    # relaxed pass disables auto-page-height + uses a wider scale window.
    try:
        return _verovio_svg(s, relaxed=True)
    except Exception as e2:
        print(f"[render_svg] verovio relaxed failed: {type(e2).__name__}: {e2}",
              file=sys.stderr)
    # Pass 3: music21 + LilyPond bridge (rarely installed in env).
    try:
        from music21 import environment
        env = environment.Environment()
        ms = env.get("musicxmlPath") or env.get("musescoreDirectPNGPath")
        lp = env.get("lilypondPath")
        if ms or lp:
            tmp = s.write("musicxml.svg")
            return Path(tmp).read_text()
    except Exception as e3:
        print(f"[render_svg] lilypond bridge failed: {type(e3).__name__}: {e3}",
              file=sys.stderr)
    # All engraved-notation paths failed. Surface a clear banner so the user
    # can see what to do, not the silent piano-roll fallback.
    return _engrave_failed_svg(len(notes), bpm)


def _verovio_svg(s: stream.Stream, relaxed: bool = False) -> str:
    """Render music21 Stream via Verovio's MusicXML loader. Returns first page SVG.

    Goes through `loadFile(temp_path)` instead of `loadData(string)` because the
    string-buffer path silently rejects MusicXML under Streamlit's threaded
    runtime (returns False from loadData even on well-formed input that the
    CLI accepts). The file path is robust across runtimes.

    Raises on any error so render_svg falls back."""
    import tempfile
    import verovio
    musicxml = write_musicxml(s)
    tk = verovio.toolkit()
    if relaxed:
        tk.setOptions({
            "scale": 35, "pageHeight": 4000, "pageWidth": 2100,
            "adjustPageHeight": True, "adjustPageWidth": True,
            "footer": "none", "header": "none",
            "breaks": "auto",
            "spacingLinear": 0.25, "spacingNonLinear": 0.6,
        })
    else:
        tk.setOptions({
            "scale": 40, "pageHeight": 2970, "pageWidth": 2100,
            "adjustPageHeight": True, "adjustPageWidth": False,
            "footer": "none", "header": "none",
        })
    # Try file-path load first (robust under Streamlit), then in-memory.
    loaded = False
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".musicxml", encoding="utf-8", delete=False,
    ) as tf:
        tf.write(musicxml)
        tmp_path = tf.name
    try:
        loaded = bool(tk.loadFile(tmp_path))
        if not loaded:
            loaded = bool(tk.loadData(musicxml))
        if not loaded:
            raise RuntimeError("verovio could not load musicxml (file + data both failed)")
        n_pages = tk.getPageCount()
        if n_pages < 1:
            raise RuntimeError("verovio rendered no pages")
        return tk.renderToSVG(1)
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _engrave_failed_svg(n_notes: int, bpm: float) -> str:
    """Banner SVG shown when every Verovio path fails on a non-empty note set.
    Replaces the silent piano-roll fallback so the UI tells the user what
    happened and what to do next (re-run, try medium/hard mode, check audio)."""
    msg1 = "Score engraving failed"
    msg2 = f"transcription returned {n_notes} notes at BPM {bpm:.0f}, but"
    msg3 = "Verovio could not lay out the notation. Try a different mode"
    msg4 = "(soft / medium / hard) or re-record with a clearer melody."
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="180" '
        'viewBox="0 0 700 180">\n'
        '<rect width="100%" height="100%" fill="#fff6f0" stroke="#d97b3a" '
        'stroke-width="2"/>\n'
        f'<text x="20" y="44" font-family="sans-serif" font-size="20" '
        f'fill="#a04020" font-weight="bold">{sx.escape(msg1)}</text>\n'
        f'<text x="20" y="84" font-family="sans-serif" font-size="13" '
        f'fill="#333">{sx.escape(msg2)}</text>\n'
        f'<text x="20" y="108" font-family="sans-serif" font-size="13" '
        f'fill="#333">{sx.escape(msg3)}</text>\n'
        f'<text x="20" y="132" font-family="sans-serif" font-size="13" '
        f'fill="#333">{sx.escape(msg4)}</text>\n'
        '</svg>'
    )


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
