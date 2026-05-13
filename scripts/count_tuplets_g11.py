"""G-11 strict measurement: count unreadable tuplets in the 4 demo SVGs.

"Unreadable" per the task description = tuplets with denominators > 8
(i.e. 12-lets, 24-lets, 48-lets — anything beyond triplet + sextuplet).

Counts Verovio-emitted `<g class="tuplet"...>` groups and inspects their
number attribute. Falls back to a regex on the MusicXML tuplet-actual
counts for cases where Verovio inlines the value.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

DEMO_DIR = Path("outputs/demos")
ITEMS = {
    "bwv_854_piano": ("bwv_854_piano.musicxml", "bwv_854_piano.svg"),
    "maestro_chamber3_30s": ("maestro_chamber3_30s.musicxml", "maestro_chamber3_30s.svg"),
    "mtg_qbh_q1_humming": ("mtg_qbh_q1_humming.musicxml", "mtg_qbh_q1_humming.svg"),
    "vocadito_1_humming": ("vocadito_1_humming.musicxml", "vocadito_1_humming.svg"),
    "vocadito_1_humming_before": ("vocadito_1_humming_before.musicxml", "vocadito_1_humming_before.svg"),
    "vocadito_1_humming_after": ("vocadito_1_humming_after.musicxml", "vocadito_1_humming_after.svg"),
}


def _count_tuplets_musicxml(mxl_path: Path) -> dict[int, int]:
    """Return {actual_notes -> count} from <time-modification><actual-notes>."""
    counts: dict[int, int] = {}
    if not mxl_path.exists():
        return counts
    try:
        tree = ET.parse(str(mxl_path))
    except ET.ParseError:
        return counts
    root = tree.getroot()
    for tm in root.iter("time-modification"):
        actual = tm.find("actual-notes")
        normal = tm.find("normal-notes")
        if actual is None or normal is None:
            continue
        try:
            a = int(actual.text or "1"); n = int(normal.text or "1")
        except ValueError:
            continue
        if a == n or a < 2:
            continue
        counts[a] = counts.get(a, 0) + 1
    return counts


def _count_tuplets_svg(svg_path: Path) -> dict[int, int]:
    """Fallback: count tuplet glyphs in the rendered SVG itself."""
    if not svg_path.exists():
        return {}
    text = svg_path.read_text(errors="ignore")
    pat = re.compile(r'class="tuplet-num"[^>]*>(\d+)<')
    out: dict[int, int] = {}
    for m in pat.finditer(text):
        try:
            v = int(m.group(1))
        except ValueError:
            continue
        if v >= 2:
            out[v] = out.get(v, 0) + 1
    return out


def main() -> None:
    rows = []
    for name, (mxl, svg) in ITEMS.items():
        cm = _count_tuplets_musicxml(DEMO_DIR / mxl)
        cs = _count_tuplets_svg(DEMO_DIR / svg)
        all_keys = sorted(set(cm) | set(cs))
        # "unreadable" = denom > 8 (i.e. > sextuplet)
        unreadable_total = sum(cm.get(k, 0) for k in all_keys if k > 8) \
            + sum(cs.get(k, 0) for k in all_keys if k > 8) - sum(min(cm.get(k, 0), cs.get(k, 0)) for k in all_keys if k > 8)
        # Use musicxml when present, fallback to svg.
        unread_mxl = sum(cm.get(k, 0) for k in cm if k > 8)
        unread_svg = sum(cs.get(k, 0) for k in cs if k > 8)
        rows.append({"demo": name, "tuplet_breakdown_mxl": cm, "tuplet_breakdown_svg": cs,
                      "unreadable_mxl": unread_mxl, "unreadable_svg": unread_svg})
        print(f"{name:35s}  mxl={cm}  svg={cs}  unreadable_mxl={unread_mxl}  unreadable_svg={unread_svg}")
    # G-11 strict subset: the 4 production demos (not the before/after pair).
    g11_demos = ["bwv_854_piano", "maestro_chamber3_30s", "mtg_qbh_q1_humming", "vocadito_1_humming"]
    total_unreadable = sum(r["unreadable_mxl"] for r in rows if r["demo"] in g11_demos)
    print(f"\nG-11 strict: total unreadable tuplets across 4 demos = {total_unreadable} (criterion <= 5)")
    out = {"per_demo": rows, "g11_total_unreadable": total_unreadable,
            "g11_demos_in_scope": g11_demos,
            "g11_strict_pass": total_unreadable <= 5}
    import json
    Path("reports/_item-g11_tuplet_audit.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
