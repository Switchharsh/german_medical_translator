#!/usr/bin/env python3
"""Render the round-trip figures as PNGs for the thesis.

Usage:
    python3 scripts/make_figures.py results/roundtrip_<id> [-o figures/]

Reads ``roundtrip_steps.csv`` (written by ``scripts/roundtrip/collect.py``) and
writes one PNG per figure at 300 dpi.

Colours come from a colourblind-safe categorical palette; where more series are
on screen than that palette can separate, the figures fall back to small
multiples (one panel, one accent hue) or to shape-plus-label encoding rather
than inventing extra hues.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ACCENT = "#2a78d6"
ACCENT_2 = "#eb6834"
CONTEXT = "#d8d7d0"
SEQ_LIGHT = "#86b6ef"
SEQ_DARK = "#1c5cab"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

API_MODELS = {"glm-5.2", "MiniMax-M3", "DeepSeek-V4-Flash"}
CONTROL = "identity"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titleweight": "600",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def load(csv_path: Path) -> dict[str, dict]:
    """Per-model curves, split by direction. English steps are the odd ones."""
    rows = list(csv.DictReader(csv_path.open()))
    grouped: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for row in rows:
        grouped[row["model"]][int(row["step"])] = row

    def number(value: str) -> float | None:
        return None if value in ("—", "") else float(value)

    out: dict[str, dict] = {}
    for model, steps in grouped.items():
        english = [s for s in sorted(steps) if steps[s]["direction"].endswith("->en")]
        german = [s for s in sorted(steps) if not steps[s]["direction"].endswith("->en")]
        out[model] = {
            "bleu": [number(steps[s]["bleu"]) for s in english],
            "chrf": [number(steps[s]["chrf"]) for s in english],
            "crit_en": [float(steps[s]["crit_rate"]) * 100 for s in english],
            "crit_de": [float(steps[s]["crit_rate"]) * 100 for s in german],
            "chars": [float(steps[s]["mean_chars"]) for s in english],
        }
    return out


def ranked(data: dict[str, dict]) -> list[str]:
    """Real models, best single-pass BLEU first. The control is never plotted."""
    return sorted(
        (m for m in data if m != CONTROL), key=lambda m: -data[m]["bleu"][0]
    )


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _legend_above(ax, handles):
    """Put the key in the margin above the plot.

    Every in-axes position collided with something on at least one of these
    figures (the bottom-right corner sat on nllb's bars and on opus's label),
    and a legend that has to be dodged is worse than one that costs a little
    vertical space.
    """
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=len(handles), frameon=False, fontsize=8,
              borderaxespad=0, handletextpad=0.6, columnspacing=1.8)


# Candidate label offsets in points, tried in order: right, left, above, below,
# then the diagonals. The first placement that collides with nothing already
# placed wins.
_LABEL_OFFSETS = [
    (8, -3, "left", "center"), (-8, -3, "right", "center"),
    (0, 9, "center", "bottom"), (0, -11, "center", "top"),
    (8, 7, "left", "bottom"), (-8, 7, "right", "bottom"),
    (8, -13, "left", "top"), (-8, -13, "right", "top"),
]


def _place_labels(ax, points, fontsize=7.8, colour=INK):
    """Annotate ``points`` [(x, y, text)] without letting the labels overlap.

    Matplotlib has no built-in label de-confliction, and on this data three
    models sit within two BLEU of each other, so the naive fixed offset
    produced an unreadable pile. Each label's real rendered box is measured and
    the next candidate offset tried until it lands clear of the others.
    """
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Seed the obstacle list with the marks themselves: a label that clears the
    # other labels but sits on top of a data point is still unreadable.
    marker_radius = 7.0
    placed: list = []
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox.from_extents(px - marker_radius, py - marker_radius,
                                        px + marker_radius, py + marker_radius))
    obstacles = len(placed)

    for x, y, text in points:
        annotation = None
        for dx, dy, ha, va in _LABEL_OFFSETS:
            if annotation is not None:
                annotation.remove()
            annotation = ax.annotate(
                text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=fontsize, color=colour, zorder=6,
            )
            box = annotation.get_window_extent(renderer=renderer)
            axes_box = ax.get_window_extent(renderer=renderer)
            inside = (box.x0 >= axes_box.x0 - 2 and box.x1 <= axes_box.x1 + 2
                      and box.y0 >= axes_box.y0 - 2 and box.y1 <= axes_box.y1 + 2)
            if inside and not any(box.overlaps(other) for other in placed):
                break
        placed.append(annotation.get_window_extent(renderer=renderer))
    return placed[obstacles:]


def fig_curves(data: dict[str, dict], out: Path) -> None:
    """Small multiples: one panel per model, the rest in grey behind it.

    Twelve series cannot be told apart by hue on one axis, so each model gets
    its own panel and a single accent colour; the grey context lines keep the
    comparison available without needing twelve legend entries.
    """
    models = ranked(data)
    cols, cycles = 4, range(1, 11)
    rows = -(-len(models) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 2.0 * rows), sharex=True, sharey=True)
    axes = axes.ravel()

    for index, model in enumerate(models):
        ax = axes[index]
        for other in models:
            if other != model:
                ax.plot(cycles, data[other]["bleu"], color=CONTEXT, lw=0.8, zorder=1)
        series = data[model]["bleu"]
        ax.plot(cycles, series, color=ACCENT, lw=1.8, zorder=3)
        ax.scatter([1, 10], [series[0], series[-1]], s=22, color=ACCENT,
                   edgecolor="white", linewidth=1.1, zorder=4)
        drop = series[0] - series[-1]
        label = f"{model} ·API" if model in API_MODELS else model
        ax.set_title(label, loc="left", pad=3, color=INK)
        ax.text(0.02, 0.06, f"{series[0]:.1f} → {series[-1]:.1f}  (−{drop:.1f})",
                transform=ax.transAxes, fontsize=7.5, color=INK_2)
        ax.set_ylim(0, 62)
        ax.set_xticks([1, 5, 10])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        _despine(ax)

    for ax in axes[len(models):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(models) - cols):len(models)]:
        ax.set_xlabel("round-trip cycle")
    for index in range(0, len(models), cols):
        axes[index].set_ylabel("BLEU")

    fig.suptitle("BLEU against the human reference across 10 round trips",
                 x=0.005, ha="left", fontsize=11, weight="600", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def fig_dumbbell(data: dict[str, dict], out: Path) -> None:
    """Cycle 1 vs cycle 10 per model: one hue, two shades (before/after)."""
    models = ranked(data)[::-1]  # best at the top once the y-axis is inverted
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(models) + 1.3))

    for y, model in enumerate(models):
        first, last = data[model]["bleu"][0], data[model]["bleu"][-1]
        ax.plot([last, first], [y, y], color=MUTED, lw=1.6, zorder=1,
                solid_capstyle="round")
        ax.scatter([first], [y], s=52, color=SEQ_LIGHT, edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.scatter([last], [y], s=52, color=SEQ_DARK, edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.text(first + 1.4, y, f"−{first - last:.1f}", va="center",
                fontsize=7.5, color=INK, fontweight="600")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} ·API" if m in API_MODELS else m for m in models],
                       color=INK, fontsize=8.5)
    ax.set_xlim(0, 66)
    ax.set_xlabel("BLEU against the human reference")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    _legend_above(ax, [
        Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=SEQ_LIGHT,
               markeredgecolor="white", label="cycle 1 (single pass)"),
        Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=SEQ_DARK,
               markeredgecolor="white", label="cycle 10"),
    ])
    ax.set_title("Quality lost to ten round trips", loc="left", fontsize=11,
                 weight="600", color=INK, pad=26)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_clinical(data: dict[str, dict], out: Path) -> None:
    """Critical-error rate by direction; the tick marks the single-pass level."""
    models = ranked(data)[::-1]
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 0.46 * len(models) + 1.5))

    for y, model in enumerate(models):
        for offset, key, colour in ((height / 2, "crit_en", ACCENT),
                                    (-height / 2, "crit_de", ACCENT_2)):
            series = data[model][key]
            ax.barh(y + offset, series[-1], height=height * 0.92, color=colour, zorder=2)
            ax.plot([series[0], series[0]], [y + offset - height / 2, y + offset + height / 2],
                    color=INK, lw=1.6, zorder=4, solid_capstyle="round")
            ax.text(series[-1] + 1.0, y + offset, f"{series[-1]:.0f}", va="center",
                    fontsize=7.5, color=INK, fontweight="600")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} ·API" if m in API_MODELS else m for m in models],
                       color=INK, fontsize=8.5)
    ax.set_xlim(0, 74)
    ax.set_xlabel("% of the 20 reports containing a critical clinical error")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    _legend_above(ax, [
        Line2D([], [], marker="s", ls="", markersize=8, color=ACCENT, label="DE→EN"),
        Line2D([], [], marker="s", ls="", markersize=8, color=ACCENT_2, label="EN→DE"),
        Line2D([], [], color=INK, lw=1.6, label="cycle-1 level (bar = cycle 10)"),
    ])
    ax.set_title("Critical clinical errors after ten round trips", loc="left",
                 fontsize=11, weight="600", color=INK, pad=26)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_bleu_vs_clinical(data: dict[str, dict], out: Path) -> None:
    """The central claim: surface quality does not predict clinical safety.

    Every point is one model at cycle 1. Hue is not used to separate models —
    with twelve of them no palette could — so identity is carried by the text
    label, and marker shape distinguishes hosted from local systems.
    """
    models = ranked(data)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for model in models:
        x = data[model]["bleu"][0]
        y = data[model]["crit_en"][0]
        hosted = model in API_MODELS
        ax.scatter([x], [y], s=74 if hosted else 62,
                   marker="D" if hosted else "o",
                   color=ACCENT if hosted else "white",
                   edgecolor=ACCENT, linewidth=1.6, zorder=3)

    ax.set_xlabel("BLEU on a single pass  (higher is better)")
    ax.set_ylabel("% of reports with a critical clinical error  (lower is better)")
    ax.set_xlim(18, 66)
    ax.set_ylim(0, 60)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    _legend_above(ax, [
        Line2D([], [], marker="D", ls="", markersize=7, color=ACCENT, label="hosted API"),
        Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor="white",
               markeredgecolor=ACCENT, label="local model"),
    ])
    ax.set_title("Surface quality does not predict clinical safety", loc="left",
                 fontsize=11, weight="600", color=INK, pad=26)
    _place_labels(ax, [(data[m]["bleu"][0], data[m]["crit_en"][0], m) for m in models])
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_output_length(data: dict[str, dict], out: Path) -> None:
    """Mean output length per cycle — the under-translation control.

    A model that emits less text trips fewer detectors without being safer, so
    the clinical numbers can only be read alongside this.
    """
    models = ranked(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cycles = range(1, 11)

    labels = []
    for model in models:
        series = data[model]["chars"]
        # A 1-character difference is noise; require a real contraction.
        shrinks = series[-1] < series[0] * 0.95
        ax.plot(cycles, series, color=ACCENT_2 if shrinks else CONTEXT,
                lw=1.9 if shrinks else 1.1, zorder=3 if shrinks else 1)
        if shrinks or series[-1] > 780 or series[0] < 580:
            labels.append((10, series[-1], model))

    ax.set_xlabel("round-trip cycle")
    ax.set_ylabel("mean output length (characters)")
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlim(0.8, 11.6)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    _legend_above(ax, [
        Line2D([], [], color=ACCENT_2, lw=1.9, label="shrinks by >5% (under-translates)"),
        Line2D([], [], color=CONTEXT, lw=1.1, label="grows or holds"),
    ])
    ax.set_title("Output length across cycles", loc="left", fontsize=11,
                 weight="600", color=INK, pad=26)
    _place_labels(ax, labels)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


FIGURES = {
    "fig1_roundtrip_curves.png": fig_curves,
    "fig2_quality_lost.png": fig_dumbbell,
    "fig3_clinical_errors.png": fig_clinical,
    "fig4_bleu_vs_clinical.png": fig_bleu_vs_clinical,
    "fig5_output_length.png": fig_output_length,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("figures"))
    args = parser.parse_args()

    csv_path = args.run_dir / "roundtrip_steps.csv"
    if not csv_path.exists():
        raise SystemExit(f"no roundtrip_steps.csv in {args.run_dir} — run collect.py first")

    data = load(csv_path)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES.items():
        builder(data, args.output / name)
        print(f"wrote {args.output / name}")
    print(f"\n{len(ranked(data))} models plotted, control '{CONTROL}' excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
