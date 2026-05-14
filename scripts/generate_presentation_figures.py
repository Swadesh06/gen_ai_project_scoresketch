"""W-3: Generate seven presentation figures for the demo-day slide deck.

Produces:
  F-1 outputs/figures/F1_metric_trajectory.png       (ASAP MV2H + Voc F1 across phases)
  F-2 outputs/figures/F2_mv2h_subaxes.png            (pre/post Phase G grouped bar)
  F-3 outputs/figures/F3_strict_pass_distribution.png(17 Phase G items donut)
  F-4 outputs/figures/F4_asap_per_piece.png          (9 ASAP pieces bar)
  F-5 outputs/figures/F5_g4_ablation.png             (Vocadito G-4 ablation bar)
  F-6 outputs/figures/F6_pipeline_architecture.png   (matplotlib boxes diagram)
  F-7 outputs/figures/F7_demo_before_after.png       (2x2 SVG-to-PNG comparison)

All 300 DPI landscape. SVG to PNG via rsvg-convert subprocess (cairosvg not
in env). Pure-CPU run, ~5 min wall.
"""
from __future__ import annotations
import io
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "outputs" / "figures"
DEMO_DIR = REPO / "outputs" / "demos"
DPI = 300
LAND_WH = (16, 10)   # 4800x3000 px @ DPI=300; downscaled for slides
LAND_WIDE = (16, 8)

RSVG = shutil.which("rsvg-convert")
if RSVG is None:
    raise SystemExit("rsvg-convert binary not found on PATH")


def _save(fig: plt.Figure, name: str) -> Path:
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size // 1024} KB)")
    return out


# ---------------------------------------------------------------------------
# F-1: Phase-by-phase MV2H + Vocadito trajectory
# ---------------------------------------------------------------------------
def fig1_trajectory() -> None:
    phases = [
        "Phase B+1",
        "Phase D",
        "Phase E start\n(post v3.4)",
        "Phase E end\n(post F-1 octave)",
        "Phase G start",
        "Phase G end",
    ]
    asap = [np.nan, np.nan, 0.5277, 0.5492, 0.5515, 0.6151]
    voc = [0.665, 0.665, 0.666, 0.666, 0.6652, 0.6776]

    fig, ax1 = plt.subplots(figsize=LAND_WIDE)
    x = np.arange(len(phases))
    ax2 = ax1.twinx()

    ln1, = ax1.plot(
        x, asap, "o-", color="#1f77b4", linewidth=2.5, markersize=9,
        label="ASAP 9-piece MV2H (score beats)",
    )
    ln2, = ax2.plot(
        x, voc, "s--", color="#d62728", linewidth=2.5, markersize=9,
        label="Vocadito A1 noff F1 (mir_eval)",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=11)
    ax1.set_ylabel("ASAP 9-piece MV2H", fontsize=13, color="#1f77b4")
    ax2.set_ylabel("Vocadito A1 noff F1", fontsize=13, color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0.50, 0.66)
    ax2.set_ylim(0.64, 0.76)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        "HumScribe headline metrics across phases\n"
        "Phase G strict-measurement lifts: ASAP +0.0636 MV2H, Vocadito +0.0124 F1",
        fontsize=14, pad=14,
    )

    # Annotate the two strict-pass lifts
    ax1.annotate(
        "+0.0636\n(G-1+G-2 emission)",
        xy=(5, 0.6151), xytext=(4.1, 0.62),
        fontsize=11, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
    )
    ax2.annotate(
        "+0.0124\n(G-4 same-pitch)",
        xy=(5, 0.6776), xytext=(3.7, 0.673),
        fontsize=11, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
    )

    # IAA ceiling reference on the Vocadito axis
    ax2.axhline(0.740, color="#888", linestyle=":", linewidth=1.5)
    ax2.text(0.1, 0.743, "Vocadito IAA ceiling 0.740 (human agreement)",
             fontsize=10, color="#444")

    ax1.legend(handles=[ln1, ln2], loc="upper left", fontsize=11)
    _save(fig, "F1_metric_trajectory.png")


# ---------------------------------------------------------------------------
# F-2: MV2H sub-axis breakdown ASAP (pre/post Phase G)
# ---------------------------------------------------------------------------
def fig2_subaxes() -> None:
    axes_names = ["multi-pitch", "voice", "meter", "value", "harmony"]
    pre = [0.962, 0.704, 0.103, 0.989, 0.000]   # pre-Phase G ASAP baseline
    post = [0.962, 0.825, 0.303, 0.985, 0.000]  # post-Phase G (G-1+G-2)

    x = np.arange(len(axes_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=LAND_WIDE)
    b1 = ax.bar(x - width / 2, pre, width, label="pre Phase G", color="#aec7e8")
    b2 = ax.bar(x + width / 2, post, width, label="post Phase G", color="#1f77b4")

    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(
                f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=10,
            )

    # Annotations: saturated axes + harmony floor
    ax.annotate(
        "saturated", xy=(0, 0.962), xytext=(0, 1.08),
        ha="center", fontsize=10, color="#555",
        arrowprops=dict(arrowstyle="->", color="#555"),
    )
    ax.annotate(
        "saturated", xy=(3, 0.985), xytext=(3, 1.08),
        ha="center", fontsize=10, color="#555",
        arrowprops=dict(arrowstyle="->", color="#555"),
    )
    ax.annotate(
        "untouched (no harmony module)",
        xy=(4, 0.02), xytext=(4, 0.30),
        ha="center", fontsize=10, color="#a04",
        arrowprops=dict(arrowstyle="->", color="#a04"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(axes_names, fontsize=12)
    ax.set_ylabel("sub-score (MV2H)", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "ASAP 9-piece MV2H sub-axes — Phase G emission work moved voice + meter\n"
        "(harmony axis is zero because no chord recognition ships)",
        fontsize=14, pad=14,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "F2_mv2h_subaxes.png")


# ---------------------------------------------------------------------------
# F-3: Strict-pass distribution donut for 17 Phase G items
# ---------------------------------------------------------------------------
def fig3_strict_pass() -> None:
    groups = [
        ("Full strict pass\n(G-4, G-7)", 2, "#2ca02c"),
        ("Partial pass\n(G-1 ASAP, G-2 ASAP)", 2, "#bcbd22"),
        ("Published failed to transfer\n(G-5, G-6, G-8, G-10)", 4, "#999"),
        ("Corpus pathology\n(G-3, G-9, G-11)", 3, "#bbb"),
        ("Infrastructure blocked\n(G-13, G-14, G-15)", 3, "#ccc"),
        ("Ensemble undersized\n(G-12)", 1, "#ddd"),
        ("Per-spec artifact\n(G-16, G-17)", 2, "#1f77b4"),
    ]
    labels = [g[0] for g in groups]
    sizes = [g[1] for g in groups]
    colors = [g[2] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 9))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        autopct=lambda p: f"{int(p * 17 / 100 + 0.5)}",
        pctdistance=0.78,
        textprops=dict(fontsize=11),
    )
    for t in autotexts:
        t.set_fontsize(13)
        t.set_color("black")
        t.set_weight("bold")

    ax.set_title(
        "Phase G strict-measurement outcomes (17 items)\n"
        "2 full passes  -  2 partial  -  11 honest discards  -  2 per-spec artifacts",
        fontsize=14, pad=18,
    )

    ax.text(0, 0, "17", ha="center", va="center", fontsize=42, weight="bold")
    ax.text(0, -0.13, "items", ha="center", va="center", fontsize=14)

    _save(fig, "F3_strict_pass_distribution.png")


# ---------------------------------------------------------------------------
# F-4: Per-piece ASAP MV2H (pre vs post Phase G), 9 pieces
# ---------------------------------------------------------------------------
def fig4_per_piece() -> None:
    # Pre-Phase-G baseline (G-1/G-2 off) + meter contribution per piece
    pieces = [
        "Bach BWV 846", "Bach BWV 848", "Bach BWV 854", "Bach BWV 856",
        "Bach BWV 857", "Beethoven 21-1", "Schumann Toccata",
        "Chopin Berceuse", "Liszt Sonata",
    ]
    # Pre/post ASAP MV2H means from per-piece data in item-g2.md + Phase E logs.
    # Pre is the "baseline (voices=[0]*n, meter=uniform)" state.
    # Post adds G-1 voice 0.825 + G-2 meter 0.303 (both 9-piece means).
    pre = [0.4830, 0.5263, 0.6101, 0.4588, 0.6224, 0.5800, 0.5670, 0.5261, 0.4752]
    post = [
        0.6072, 0.6086, 0.6713, 0.5586, 0.6722,
        0.6207, 0.6452, 0.5312, 0.5215,
    ]

    x = np.arange(len(pieces))
    width = 0.36
    fig, ax = plt.subplots(figsize=LAND_WIDE)
    b1 = ax.bar(x - width / 2, pre, width, label="pre Phase G", color="#aec7e8")
    b2 = ax.bar(x + width / 2, post, width, label="post Phase G", color="#1f77b4")

    # Highlight BWV 856 (the F-1 octave + meter +0.100 piece)
    b1[3].set_color("#ff9896")
    b2[3].set_color("#d62728")
    b2[3].set_edgecolor("black")
    b2[3].set_linewidth(2)
    # Highlight Chopin Berceuse — the F-1 detector's miss case
    b1[7].set_color("#dddddd")
    b2[7].set_color("#888888")

    ax.annotate(
        "+0.100 from F-1 octave\n(Phase E)",
        xy=(3, post[3]), xytext=(3.5, 0.36),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4),
    )
    ax.annotate(
        "F-1 detector miss\n(3x tempo error)",
        xy=(7, post[7]), xytext=(6.6, 0.32),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="#666", lw=1.4),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pieces, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("MV2H", fontsize=13)
    ax.set_ylim(0, 0.80)
    ax.set_title(
        "ASAP per-piece MV2H — pre vs post Phase G (9 pieces)\n"
        "BWV 856 picks up +0.100 from F-1 octave sanity; Chopin Berceuse stays flat",
        fontsize=14, pad=14,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "F4_asap_per_piece.png")


# ---------------------------------------------------------------------------
# F-5: Vocadito G-4 ablation
# ---------------------------------------------------------------------------
def fig5_g4_ablation() -> None:
    states = [
        "baseline\n(all post off)",
        "G-4 alone\n(merge same-pitch)",
        "G-5 alone\n(median smooth)",
        "G-4 + G-5 + G-6\n(all on)",
    ]
    f1 = [0.6652, 0.6776, 0.6520, 0.6587]
    colors = ["#aec7e8", "#2ca02c", "#d62728", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=LAND_WIDE)
    bars = ax.bar(states, f1, color=colors, edgecolor="black", linewidth=1.2)
    for r, v in zip(bars, f1):
        ax.annotate(
            f"{v:.4f}",
            xy=(r.get_x() + r.get_width() / 2, v),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=11, weight="bold",
        )

    # Strict threshold line at 0.67
    ax.axhline(0.67, color="#a04", linestyle="--", linewidth=1.8)
    ax.text(
        3.45, 0.6715, "strict threshold 0.67",
        fontsize=10, color="#a04", ha="right",
    )

    ax.set_ylabel("Vocadito A1 noff F1 (mir_eval)", fontsize=13)
    ax.set_ylim(0.62, 0.70)
    ax.set_title(
        "Vocadito G-4 ablation (40-clip A1)\n"
        "G-4 alone clears the strict 0.67 bar; G-5/G-6 regress it",
        fontsize=14, pad=14,
    )
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "F5_g4_ablation.png")


# ---------------------------------------------------------------------------
# F-6: Pipeline architecture diagram (six stages + Stage 7 arrange)
# ---------------------------------------------------------------------------
def fig6_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title(
        "HumScribe pipeline — 6 stages (Stage 7 optional arrange)",
        fontsize=15, pad=16,
    )

    def box(x, y, w, h, label, sub, color="#dbe7f5"):
        bx = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#222", linewidth=1.4,
        )
        ax.add_patch(bx)
        ax.text(
            x + w / 2, y + h - 0.4, label,
            ha="center", va="top", fontsize=12, weight="bold",
        )
        ax.text(
            x + w / 2, y + 0.45, sub,
            ha="center", va="bottom", fontsize=9,
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="->",
            mutation_scale=18, color="#222", linewidth=1.5,
        ))

    # Main flow on row y=4.6 (height 2.6)
    rows = [
        (0.2, 4.4, 2.0, 2.6, "Stage 0\naudio I/O",
         "ffmpeg + librosa\n22.05 kHz mono", "#dbe7f5"),
        (2.5, 4.4, 2.0, 2.6, "Stage 1\nmode gate",
         "humming / instrument /\npiano / guitar", "#dbe7f5"),
        (8.0, 6.6, 2.4, 2.0, "Stage 2-A\ninstrument",
         "YourMT3+ (piano)\nByteDance\nBasic Pitch", "#cde8c8"),
        (8.0, 2.2, 2.4, 2.0, "Stage 2-B\nhumming",
         "PESTO pitch\nCREPE periodicity\n(voicing)", "#f6d4a8"),
        (10.7, 4.4, 1.8, 2.6, "Stage 3\nnormalise",
         "post_process.py\nG-4 merge\nspell + key", "#dbe7f5"),
        (12.7, 4.4, 1.8, 2.6, "Stage 4\nbeat",
         "beat_this\nF-1 octave\nsanity", "#dbe7f5"),
        (14.7, 4.4, 1.6, 2.6, "Stage 5\nrhythm DP",
         "Cemgil-Kappen\ntpb=12,\nper-voice", "#dbe7f5"),
        (16.4, 4.4, 1.5, 2.6, "Stage 6\nrender",
         "music21\nVerovio\nSVG + MXL", "#dbe7f5"),
    ]
    for r in rows:
        box(*r)

    # Branch labels
    ax.text(6.5, 7.0, "kind in {piano, instrument}", fontsize=9, color="#246")
    ax.text(6.5, 2.0, "kind == humming", fontsize=9, color="#a40")

    # Arrows along main flow
    arrow(2.2, 5.7, 2.5, 5.7)
    arrow(4.5, 5.7, 7.95, 5.7)
    arrow(4.5, 5.7, 7.95, 3.2)
    arrow(10.4, 7.5, 10.7, 6.4)
    arrow(10.4, 3.2, 10.7, 5.0)
    arrow(12.5, 5.7, 12.7, 5.7)
    arrow(14.5, 5.7, 14.7, 5.7)
    arrow(16.3, 5.7, 16.4, 5.7)

    # Optional Stage 7 below
    box(
        13.6, 0.2, 4.2, 2.2, "Stage 7 (optional) arrangement",
        "MusicGen-Melody-Large\nC5b LoRA adapter (JSB Chorales)",
        "#f0d0e8",
    )
    arrow(15.5, 4.4, 15.5, 2.4)

    # Optional voice transformer note
    box(
        0.2, 0.4, 6.0, 1.8,
        "B76 voice transformer (auto-routes for Chopin-style)",
        "6-layer Transformer trained on 237 ASAP pieces;\n"
        "per-voice DP follows when triggered",
        "#f5f0d0",
    )

    _save(fig, "F6_pipeline_architecture.png")


# ---------------------------------------------------------------------------
# F-7: Composite demo before/after visual comparison
# ---------------------------------------------------------------------------
def _svg_to_png(svg_path: Path, width_px: int = 1800) -> Image.Image:
    cmd = [RSVG, "-w", str(width_px), "-f", "png", str(svg_path)]
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    return Image.open(io.BytesIO(out.stdout)).convert("RGBA")


def fig7_demo_before_after() -> None:
    panels = [
        (
            "MAESTRO chamber - PRE (Phase G regression)",
            DEMO_DIR / "maestro_chamber3_30s_phase_g_regression.svg",
            "tempo 154, 4x 24-let + 1x 48-let",
        ),
        (
            "MAESTRO chamber - AFTER W-1 revert",
            DEMO_DIR / "maestro_chamber3_30s.svg",
            "tempo 154 integer, 0 unreadable tuplets",
        ),
        (
            "Vocadito - BEFORE G-4",
            DEMO_DIR / "vocadito_1_humming_before.svg",
            "rapid-repeat fragmentation",
        ),
        (
            "Vocadito - AFTER G-4",
            DEMO_DIR / "vocadito_1_humming_after.svg",
            "same-pitch gaps merged",
        ),
    ]

    fig = plt.figure(figsize=(16, 10))
    for i, (title, path, sub) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1)
        if path.exists():
            try:
                img = _svg_to_png(path)
                ax.imshow(img)
                ax.set_aspect("auto")
            except Exception as e:
                ax.text(0.5, 0.5, f"render failed:\n{e}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="red")
        else:
            ax.text(0.5, 0.5, f"missing:\n{path.name}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n{sub}", fontsize=12, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#999")

    fig.suptitle(
        "Demo visual comparison: regression and G-4 wins side-by-side",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "F7_demo_before_after.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_trajectory()
    fig2_subaxes()
    fig3_strict_pass()
    fig4_per_piece()
    fig5_g4_ablation()
    fig6_pipeline()
    fig7_demo_before_after()
    print(f"\nall 7 figures written to {FIG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
