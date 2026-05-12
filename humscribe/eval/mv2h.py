"""MV2H evaluation wrapper (Phase E item 1).

Shells out to the compiled MV2H jar (third_party/MV2H/bin) and parses the
five sub-scores plus the average. Non-aligned (`-a`) DTW is the default
because our pipeline produces self-consistent timings that don't share an
absolute time base with the reference.

Usage:
    from humscribe.eval.mv2h import compute_mv2h, MV2HResult
    r = compute_mv2h(predicted_text, reference_text)
    print(r.mv2h, r.multi_pitch, r.voice, r.meter, r.value, r.harmony)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import re
import subprocess
import tempfile
from typing import Literal

from humscribe.eval.mv2h_io import (
    notes_to_mv2h_format,
    stream_to_mv2h_format,
    musicxml_to_mv2h_format,
    midi_to_mv2h_format,
)


_DEFAULT_JAR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "MV2H" / "bin"


@dataclass
class MV2HResult:
    multi_pitch: float
    voice: float
    meter: float
    value: float
    harmony: float
    mv2h: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


_LINE_RE = re.compile(
    r"(Multi-pitch|Voice|Meter|Value|Harmony|MV2H)\s*:\s*([\d.eE+-]+|NaN)"
)


def _parse_mv2h_stdout(stdout: str) -> MV2HResult:
    parts: dict[str, float] = {}
    for m in _LINE_RE.finditer(stdout):
        key = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            val = float("nan")
        parts[key] = val
    return MV2HResult(
        multi_pitch=parts.get("Multi-pitch", float("nan")),
        voice=parts.get("Voice", float("nan")),
        meter=parts.get("Meter", float("nan")),
        value=parts.get("Value", float("nan")),
        harmony=parts.get("Harmony", float("nan")),
        mv2h=parts.get("MV2H", float("nan")),
    )


def _run_mv2h_jar(gt_path: str, pred_path: str, *,
                  jar_dir: Path = _DEFAULT_JAR_DIR,
                  align: Literal["aligned", "non_aligned", "verbose"] = "non_aligned",
                  alignment_penalty: float = 1.0,
                  timeout_s: float = 120.0) -> tuple[MV2HResult, str]:
    """Run the MV2H jar; return (result, raw_stdout)."""
    java = os.environ.get("MV2H_JAVA", "java")
    cmd = [java, "-cp", str(jar_dir), "mv2h.Main",
           "-g", str(gt_path), "-t", str(pred_path)]
    if align == "non_aligned":
        cmd += ["-a", "-p", f"{alignment_penalty}"]
    elif align == "verbose":
        cmd += ["-A", "-p", f"{alignment_penalty}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mv2h jar exited {proc.returncode}: {proc.stderr.strip()[:300]}\n"
            f"stdout head: {proc.stdout[:300]}")
    return _parse_mv2h_stdout(proc.stdout), proc.stdout


def compute_mv2h(predicted_text: str, reference_text: str, *,
                 align: Literal["aligned", "non_aligned", "verbose"] = "non_aligned",
                 alignment_penalty: float = 1.0,
                 jar_dir: Path = _DEFAULT_JAR_DIR) -> MV2HResult:
    """Run MV2H on two MV2H-format strings."""
    with tempfile.TemporaryDirectory() as td:
        gt_p = Path(td) / "gt.txt"
        pr_p = Path(td) / "pred.txt"
        gt_p.write_text(reference_text)
        pr_p.write_text(predicted_text)
        res, _ = _run_mv2h_jar(str(gt_p), str(pr_p), jar_dir=jar_dir,
                               align=align, alignment_penalty=alignment_penalty)
    return res


def compute_mv2h_from_streams(pred_stream, ref_stream, *,
                              align: Literal["aligned", "non_aligned"] = "non_aligned",
                              tatums_per_beat: int = 4,
                              jar_dir: Path = _DEFAULT_JAR_DIR) -> MV2HResult:
    pred_text = stream_to_mv2h_format(pred_stream, tatums_per_beat=tatums_per_beat)
    ref_text = stream_to_mv2h_format(ref_stream, tatums_per_beat=tatums_per_beat)
    return compute_mv2h(pred_text, ref_text, align=align, jar_dir=jar_dir)


def compute_mv2h_from_files(pred_path: str, ref_path: str, *,
                            align: Literal["aligned", "non_aligned"] = "non_aligned",
                            tatums_per_beat: int = 4,
                            jar_dir: Path = _DEFAULT_JAR_DIR) -> MV2HResult:
    """Convert each file (.mid/.midi/.xml/.musicxml/.mxl/.txt) and evaluate."""
    pred_text = _file_to_mv2h_text(pred_path, tatums_per_beat=tatums_per_beat)
    ref_text = _file_to_mv2h_text(ref_path, tatums_per_beat=tatums_per_beat)
    return compute_mv2h(pred_text, ref_text, align=align, jar_dir=jar_dir)


def _file_to_mv2h_text(path: str | Path, *, tatums_per_beat: int = 4) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".txt":
        return p.read_text()
    if suf in (".mid", ".midi"):
        return midi_to_mv2h_format(p, tatums_per_beat=tatums_per_beat)
    if suf in (".xml", ".musicxml", ".mxl"):
        return musicxml_to_mv2h_format(p, tatums_per_beat=tatums_per_beat)
    raise ValueError(f"unsupported MV2H input format: {suf}")
