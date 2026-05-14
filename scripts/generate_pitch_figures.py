"""W-2: Four standalone slide figures for the 4-minute presentation.

Produces:
  outputs/figures/S1_concept.png          — audio -> pipeline -> score
  outputs/figures/S2_subaxis_headroom.png — MV2H sub-axes pre/post Phase G
  outputs/figures/S3_pipeline.png         — single-row 7-stage pipeline
  outputs/figures/S4_before_after.png     — Vocadito BEFORE/AFTER G-4 only

Each PNG is 300 DPI, 16:9, standalone. No 2x2 grids. SVG -> PNG via
cairosvg (now in env after setup_linux.sh installed it).
"""
from __future__ import annotations
import io
from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "outputs" / "figures"
DEMO_DIR = REPO / "outputs" / "demos"
DPI = 300

# 16:9 figure sizes at 300 DPI:
#   (16, 9)  -> 4800 x 2700 px (very wide)
#   (16, 10) -> 4800 x 3000 px (just slightly taller, easier annotations)
SIZE_169 = (16, 9)
SIZE_1610 = (16, 10)


def _save(fig: plt.Figure, name: str) -> Path:
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------------------------------------------------------------------
# S1 — concept diagram (audio -> pipeline -> score)
# ---------------------------------------------------------------------------
def s1_concept() -> None:
    fig, ax = plt.subplots(figsize=SIZE_169)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("HumScribe: Audio  →  Score", fontsize=24, pad=20, weight="bold")

    # Three big boxes, centered around y=5
    box_y = 4.0
    box_h = 3.2

    def box(x, w, label, sub, color):
        bx = FancyBboxPatch(
            (x, box_y), w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.2",
            facecolor=color, edgecolor="#222", linewidth=2.5,
        )
        ax.add_patch(bx)
        ax.text(
            x + w / 2, box_y + box_h - 0.65, label,
            ha="center", va="center", fontsize=22, weight="bold",
        )
        ax.text(
            x + w / 2, box_y + 0.65, sub,
            ha="center", va="center", fontsize=14,
        )

    # Box 1: Audio input
    box(0.5, 4.5, "Audio Input", "instrument  or  hum", "#dbe7f5")
    # Stylized waveform inside box 1
    import numpy as np
    t = np.linspace(0, 1, 200)
    wave = 0.5 * np.sin(2 * np.pi * 5 * t) * np.exp(-1.5 * (t - 0.5) ** 2 / 0.15)
    ax.plot(0.7 + 4.1 * t, box_y + 1.7 + 0.5 * wave, color="#1f77b4", linewidth=1.5)

    # Box 2: System
    box(6.0, 4.0, "HumScribe", "6-stage pipeline\n+ optional Stage 7 arrange", "#fff4c8")

    # Box 3: Score
    box(11.0, 4.5, "Score Output", "MIDI  /  MusicXML  /  SVG", "#cde8c8")
    # Stylized staff inside box 3
    staff_x0 = 11.3
    staff_x1 = 15.2
    for i in range(5):
        ax.plot([staff_x0, staff_x1],
                [box_y + 1.5 + i * 0.18, box_y + 1.5 + i * 0.18],
                color="#222", linewidth=1.0)
    # A few note glyphs (filled circles + stems)
    for nx, ny in [(11.9, 1.6), (12.4, 1.9), (12.9, 1.7), (13.4, 2.05),
                   (14.0, 1.85), (14.6, 1.95)]:
        ax.add_patch(plt.Circle((nx, box_y + ny), 0.10, color="#222"))
        ax.plot([nx + 0.10, nx + 0.10],
                [box_y + ny, box_y + ny + 0.7],
                color="#222", linewidth=1.5)

    # Arrows between boxes
    arrow_kw = dict(arrowstyle="-|>", mutation_scale=32,
                    color="#222", linewidth=3)
    ax.add_patch(FancyArrowPatch((5.05, 5.6), (5.95, 5.6), **arrow_kw))
    ax.add_patch(FancyArrowPatch((10.05, 5.6), (10.95, 5.6), **arrow_kw))

    # Mode selector below
    ax.text(8.0, 2.6, "Mode:  soft   |   medium   |   hard",
            ha="center", va="center", fontsize=20, weight="bold", color="#444")
    ax.text(8.0, 1.9,
            "soft = audio only       medium = + BPM       hard = + key + meter",
            ha="center", va="center", fontsize=12, color="#666", style="italic")

    _save(fig, "S1_concept.png")


# ---------------------------------------------------------------------------
# S2 — MV2H sub-axis grouped bar with 4 annotations
# ---------------------------------------------------------------------------
def s2_subaxis() -> None:
    import numpy as np
    axes_names = ["multi-pitch", "voice", "meter", "value", "harmony"]
    pre = [0.962, 0.704, 0.103, 0.989, 0.000]
    post = [0.962, 0.825, 0.303, 0.985, 0.000]

    fig, ax = plt.subplots(figsize=SIZE_169)
    x = np.arange(len(axes_names))
    width = 0.36
    b1 = ax.bar(x - width / 2, pre, width, label="pre Phase G",
                color="#bbbbbb", edgecolor="#444", linewidth=1.2)
    b2 = ax.bar(x + width / 2, post, width, label="post Phase G",
                color="#2ca02c", edgecolor="#1a5e1a", linewidth=1.2)

    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=12, weight="bold",
            )

    # Required annotations:
    ax.annotate("saturated", xy=(0, 0.962), xytext=(0, 1.10),
                ha="center", fontsize=14, color="#444", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    ax.annotate("+0.121", xy=(1 + width / 2, 0.825), xytext=(1.55, 0.95),
                fontsize=15, color="#1a5e1a", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#1a5e1a", lw=2))
    ax.annotate("+0.200", xy=(2 + width / 2, 0.303), xytext=(2.55, 0.55),
                fontsize=15, color="#1a5e1a", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#1a5e1a", lw=2))
    ax.annotate("saturated", xy=(3, 0.989), xytext=(3, 1.10),
                ha="center", fontsize=14, color="#444", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
    ax.annotate("Phase H target",
                xy=(4, 0.02), xytext=(4, 0.30),
                ha="center", fontsize=14, color="#a04", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#a04", lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels(axes_names, fontsize=16)
    ax.set_ylabel("MV2H sub-score (0 to 1)", fontsize=15)
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "ASAP 9-piece MV2H sub-axes:  Phase G emission work moved voice + meter",
        fontsize=17, pad=14,
    )
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)

    _save(fig, "S2_subaxis_headroom.png")


# ---------------------------------------------------------------------------
# S3 — single-row 7-stage pipeline (Stage 7 dotted box below)
# ---------------------------------------------------------------------------
def s3_pipeline() -> None:
    fig, ax = plt.subplots(figsize=SIZE_169)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title(
        "Six-stage pipeline  +  optional generative Stage 7",
        fontsize=20, weight="bold", pad=14,
    )

    box_y = 4.5
    box_h = 2.4

    def box(x, w, label, sub, color="#dbe7f5"):
        bx = FancyBboxPatch(
            (x, box_y), w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=color, edgecolor="#222", linewidth=1.6,
        )
        ax.add_patch(bx)
        ax.text(
            x + w / 2, box_y + box_h - 0.36, label,
            ha="center", va="top", fontsize=14, weight="bold",
        )
        ax.text(
            x + w / 2, box_y + 0.85, sub,
            ha="center", va="center", fontsize=10,
        )

    # Seven stages
    stages = [
        ("Audio I/O",                "ffmpeg + librosa\n22.05 kHz mono",                                    "#dbe7f5"),
        ("Mode Gate",                "soft / medium /\nhard",                                                "#dbe7f5"),
        ("Transcribe",               "Piano: YourMT3+ / ByteDance\nHum: PESTO + CREPE + HMM",                "#cde8c8"),
        ("Normalize",                "post-process,\nG-4 merge",                                             "#dbe7f5"),
        ("Beat",                     "beat_this  +\nF-1 octave sanity",                                      "#dbe7f5"),
        ("Rhythm DP",                "Cemgil-Kappen  +\nB76 voice tracker",                                  "#dbe7f5"),
        ("Score",                    "music21  +\nVerovio (SVG)",                                            "#dbe7f5"),
    ]
    widths = [2.4, 2.6, 4.2, 2.6, 2.6, 3.4, 2.8]
    xs = []
    gap = 0.18
    cur = 0.4
    for w in widths:
        xs.append(cur)
        cur += w + gap

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=20, color="#222", linewidth=1.8)
    for i, ((label, sub, color), x, w) in enumerate(zip(stages, xs, widths)):
        box(x, w, label, sub, color)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.005, box_y + box_h / 2),
                (xs[i + 1] - 0.005, box_y + box_h / 2),
                **arrow_kw,
            ))

    # Dotted Stage 7 below Score box
    s7_x = xs[-1] + widths[-1] - 5.5
    s7 = FancyBboxPatch(
        (s7_x, 0.6), 5.5, 2.6,
        boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor="#f0d0e8", edgecolor="#a040a0", linewidth=1.8,
        linestyle="--",
    )
    ax.add_patch(s7)
    ax.text(s7_x + 5.5 / 2, 3.0 - 0.3, "Stage 7 (optional) — Arrange",
            ha="center", va="top", fontsize=14, weight="bold")
    ax.text(s7_x + 5.5 / 2, 1.5,
            "MusicGen-Melody-Large\n+ C5b r=64 LoRA (JSB Chorales)",
            ha="center", va="center", fontsize=11)
    # Arrow from Score down to Stage 7
    ax.add_patch(FancyArrowPatch(
        (xs[-1] + widths[-1] / 2, box_y),
        (xs[-1] + widths[-1] / 2, 3.2),
        arrowstyle="-|>", mutation_scale=20,
        color="#a040a0", linewidth=1.8, linestyle="--",
    ))

    _save(fig, "S3_pipeline.png")


# ---------------------------------------------------------------------------
# S4 — Vocadito BEFORE vs AFTER G-4 only (no MAESTRO)
# ---------------------------------------------------------------------------
def s4_before_after() -> None:
    before_svg = DEMO_DIR / "vocadito_1_humming_before.svg"
    after_svg = DEMO_DIR / "vocadito_1_humming_after.svg"
    if not before_svg.exists():
        raise SystemExit(f"missing {before_svg}")
    if not after_svg.exists():
        raise SystemExit(f"missing {after_svg}")

    def svg_to_pil(path: Path, width_px: int = 1800) -> Image.Image:
        png_bytes = cairosvg.svg2png(
            url=str(path), output_width=width_px,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    img_before = svg_to_pil(before_svg)
    img_after = svg_to_pil(after_svg)

    fig = plt.figure(figsize=SIZE_169, facecolor="white")
    fig.suptitle(
        "Vocadito clip 1 — same humming, two post-processing states",
        fontsize=18, weight="bold", y=0.96,
    )

    ax_l = fig.add_subplot(1, 2, 1)
    ax_l.imshow(img_before)
    ax_l.set_title("Before G-4:  13 triplet brackets, 2x 12-lets",
                   fontsize=15, weight="bold", color="#a04", pad=10)
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    for spine in ax_l.spines.values():
        spine.set_visible(True)
        spine.set_color("#a04")
        spine.set_linewidth(2)

    ax_r = fig.add_subplot(1, 2, 2)
    ax_r.imshow(img_after)
    ax_r.set_title("After G-4:  zero tuplets, same melody",
                   fontsize=15, weight="bold", color="#1a5e1a", pad=10)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    for spine in ax_r.spines.values():
        spine.set_visible(True)
        spine.set_color("#1a5e1a")
        spine.set_linewidth(2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "S4_before_after.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s1_concept()
    s2_subaxis()
    s3_pipeline()
    s4_before_after()
    print(f"\nall 4 slide figures written to {FIG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
