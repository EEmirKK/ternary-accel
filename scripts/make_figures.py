"""Figures for the README, drawn only from the measured files in results/.

    python scripts/make_figures.py        # writes figures/<name>.svg and figures/<name>.png
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # render to files only, no window
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# One colour per build in every figure.
COLOR = {"int8": "#2a78d6", "ternary": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
BUILDS = ("int8", "ternary")
PERIOD = "17.5ns"                         # the headline clock constraint

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": INK, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "svg.hashsalt": "ternary-accel",     # fixed SVG ids, so the files only change when the figures do
})


def load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text())


def energy_nj(build: str) -> float:
    """Dynamic energy per image (nJ): gate-level power over 8 Fashion-MNIST test images."""
    return load(f"asic/{build}_{PERIOD}_power.json")["energy_per_image_dynamic_j"] * 1e9


def tint(hex_color: str, amount: float) -> str:
    """Mix a colour with the surface: amount 0 = the colour, 1 = the surface."""
    c = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    s = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a + (b - a) * amount):02x}" for a, b in zip(c, s))


def shade(hex_color: str, amount: float) -> str:
    """Darken a colour towards black: amount 0 = the colour, 1 = black."""
    return "#" + "".join(f"{round(int(hex_color[i:i + 2], 16) * (1 - amount)):02x}" for i in (1, 3, 5))


def reduction(int8: float, ternary: float) -> str:
    return f"−{100 * (int8 - ternary) / int8:.0f}%"


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_area_energy() -> None:
    """Fig 1: cell area split into PE array / rest of core, and energy per image (4x4, 17.5 ns)."""
    area = {b: load(f"asic/{b}_{PERIOD}_area.json") for b in BUILDS}
    array = {b: area[b]["array_logic_area_um2"] / 1e6 for b in BUILDS}      # mm^2
    cell = {b: area[b]["logic_area_um2"] / 1e6 for b in BUILDS}
    rest = {b: cell[b] - array[b] for b in BUILDS}
    energy = {b: energy_nj(b) for b in BUILDS}

    xlim_a, xlim_e = (-0.6, 2.35), (-0.6, 1.6)       # area panel has side labels on the right
    fig, (ax_a, ax_e) = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={
        "width_ratios": [xlim_a[1] - xlim_a[0], xlim_e[1] - xlim_e[0]]})   # same bar width in both
    x = range(len(BUILDS))
    w = 0.55
    gap = dict(edgecolor=SURFACE, linewidth=1.5)    # thin surface line between touching fills

    # area: PE array at the bottom (full colour), rest of core on top (tint of the same colour)
    ax_a.bar(x, [array[b] for b in BUILDS], w, color=[COLOR[b] for b in BUILDS], **gap)
    ax_a.bar(x, [rest[b] for b in BUILDS], w, bottom=[array[b] for b in BUILDS],
             color=[tint(COLOR[b], 0.6) for b in BUILDS], **gap)
    ax_a.text(0, array["int8"] / 2, "PE array", ha="center", va="center", color="white", fontsize=9)
    ax_a.text(0, array["int8"] + rest["int8"] / 2, "rest of core", ha="center", va="center", fontsize=9)
    # how much each part shrinks, beside the ternary bar
    side = dict(ha="left", va="center", fontsize=9, color=MUTED)
    ax_a.text(1 + w / 2 + 0.05, array["ternary"] / 2,
              f"PE array {reduction(array['int8'], array['ternary'])}", **side)
    ax_a.text(1 + w / 2 + 0.05, array["ternary"] + rest["ternary"] / 2,
              f"rest of core {reduction(rest['int8'], rest['ternary'])}", **side)
    for i, b in enumerate(BUILDS):
        label = f"{cell[b]:.3f} mm²" + ("" if b == "int8" else f"  ({reduction(cell['int8'], cell[b])})")
        ax_a.text(i, cell[b], label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_a.set_title("Standard-cell area (mm²)")
    ax_a.set_ylim(0, cell["int8"] * 1.15)

    ax_e.bar(x, [energy[b] for b in BUILDS], w, color=[COLOR[b] for b in BUILDS], **gap)
    for i, b in enumerate(BUILDS):
        label = f"{energy[b]:,.0f} nJ" + ("" if b == "int8" else f"  ({reduction(energy['int8'], energy[b])})")
        ax_e.text(i, energy[b], label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_e.set_title("Dynamic energy per image (nJ)")
    ax_e.set_ylim(0, energy["int8"] * 1.15)

    for ax, xlim in ((ax_a, xlim_a), (ax_e, xlim_e)):
        ax.set_xticks(list(x), BUILDS)
        ax.set_xlim(*xlim)
        ax.locator_params(axis="y", nbins=4)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,g}")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig1_area_energy")


def fig_accuracy_energy() -> None:
    """Fig 2: Fashion-MNIST test accuracy (mean +- sample std over 3 seeds) vs energy per image.
    Fashion only: energy was measured on Fashion-MNIST images, so MNIST has no energy point."""
    runs = [json.loads(line) for line in (RESULTS / "accuracy" / "accuracy_runs.jsonl").open()]
    fashion = [r for r in runs if r["dataset"] == "fashion"]
    acc = {}
    for b in BUILDS:
        a = [100 * r["methods"]["positive"]["acc"] for r in fashion if r["precision"] == b]   # hardware's scale
        acc[b] = (statistics.mean(a), statistics.stdev(a))
    float_acc = statistics.mean(100 * r["acc_float_eval"] for r in fashion if r["precision"] == "float")
    energy = {b: energy_nj(b) for b in BUILDS}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axhline(float_acc, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(40, float_acc + 0.06, f"float (not quantised)  {float_acc:.2f}%", fontsize=9, color=MUTED, va="bottom")
    for b in BUILDS:
        m, sd = acc[b]
        ax.errorbar(energy[b], m, yerr=sd, fmt="o", markersize=8, color=COLOR[b],
                    elinewidth=1.5, capsize=4, zorder=3)
        ax.annotate(f"{b}  {m:.2f} ± {sd:.2f}%", (energy[b], m), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold")
    # the trade-off in one line, in the empty space below the points
    d_e = reduction(energy["int8"], energy["ternary"])
    d_a = acc["int8"][0] - acc["ternary"][0]
    # per-seed gaps: same recipe and seed for both precisions, so seeds pair up
    by_seed = {(r["precision"], r["seed"]): 100 * r["methods"]["positive"]["acc"]
               for r in fashion if r["precision"] in BUILDS}
    gaps = [by_seed["int8", s] - by_seed["ternary", s] for s in sorted({r["seed"] for r in fashion})]
    seeds = ("ternary lower on all " if min(gaps) > 0 else "ternary lower on some ") + \
        f"{len(gaps)} seeds ({min(gaps):.2f}–{max(gaps):.2f} pt)"
    ax.text((energy["int8"] + energy["ternary"]) / 2, 85.65,
            f"ternary vs int8: {d_e} energy, −{d_a:.1f} pt accuracy\n{seeds}",
            ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.4,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))          # hides the gridline behind it

    ax.set_title("Fashion-MNIST test accuracy vs energy per image")
    ax.set_xlabel("Dynamic energy per image (nJ)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, 1700)
    ax.set_ylim(85.5, 88.25)
    ax.spines["left"].set_visible(True)
    ax.set_xticks(range(0, 1700, 400))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,g}")
    fig.tight_layout()
    save(fig, "fig2_accuracy_energy")


def fig_size_sweep() -> None:
    """Fig 3: ternary / int8 ratio vs array size N for cell area, PE-array area and energy per image.
    Hollow markers: runs that did not pass full signoff (closes = false in sweep.json)."""
    rows = {(r["N"], r["precision"]): r for r in load("sweep/sweep.json")}
    sizes = sorted({n for n, _ in rows})
    # sweep.json's core_area_um2 is the standard-cell (logic) area, the same as Fig 1's cell area
    # Every line is a ternary/int8 ratio, so all three use the ternary orange. The base colour
    # (3.1:1 on the surface) is the lightest that stays above 3:1, so the others are darker.
    metrics = [("energy per image", "energy_nj", shade(COLOR["ternary"], 0.6), "o"),
               ("cell area", "core_area_um2", shade(COLOR["ternary"], 0.3), "s"),
               ("PE-array area", "array_area_um2", COLOR["ternary"], "^")]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(sizes[0], 1.02, "int8 = 1", fontsize=9, color=MUTED, va="bottom")
    for name, field, color, marker in metrics:
        ratio = [rows[n, "ternary"][field] / rows[n, "int8"][field] for n in sizes]
        ax.plot(sizes, ratio, color=color, linewidth=2, zorder=2)
        for n, r in zip(sizes, ratio):
            signed_off = rows[n, "int8"]["closes"] and rows[n, "ternary"]["closes"]
            ax.plot(n, r, marker=marker, markersize=8, color=color, zorder=3,
                    markerfacecolor=color if signed_off else SURFACE, markeredgewidth=1.5)
        # Labels use text colours, because the base orange is too faint for text. The marker
        # and the label's position say which line it is.
        ax.text(sizes[0] / 1.1, ratio[0], f"{ratio[0]:.2f}", ha="right", va="center", fontsize=9, color=MUTED)
        ax.text(sizes[-1] * 1.1, ratio[-1], f"{ratio[-1]:.2f}  {name}", ha="left", va="center",
                fontsize=9, color=INK, fontweight="bold")

    ax.set_title("Ternary relative to int8 as the array grows")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{n}×{n}" for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] / 1.5, sizes[-1] * 3.2)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("PE array size")
    ax.set_ylabel("ternary / int8")
    ax.spines["left"].set_visible(True)
    ax.text(sizes[-1] * 3.1, 0.03, "hollow: not fully signed off", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig3_size_sweep")


def fig_double_buffering() -> None:
    """Fig 4: PE-array utilisation, single- vs double-buffered, from the on-chip counters
    (two-layer MLP, batch of 8 Fashion-MNIST images, 4x4 array)."""
    rows = [json.loads(line) for line in (RESULTS / "sim" / "perf.jsonl").open()]
    util = {(r["precision"], r["double_buf"]): 100 * r["utilisation"] for r in rows if r["batch"] == 8}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    w = 0.6
    gap = dict(edgecolor=SURFACE, linewidth=1.5)
    ticks, tick_labels = [], []
    for g, b in enumerate(BUILDS):
        for k, db in enumerate((False, True)):
            x = 2 * g + (k - 0.5) * (w + 0.08)
            u = util[b, db]
            ax.bar(x, u, w, color=COLOR[b] if db else tint(COLOR[b], 0.6), **gap)
            label = f"{u:.0f}%" + (f"  (+{u - util[b, False]:.0f} pt)" if db else "")
            ax.text(x, u + 1.5, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
            ticks.append(x)
            tick_labels.append("double" if db else "single")
        # build name under its pair of bars: x in data units, y in axes units
        ax.text(2 * g, -0.13, b, transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold")

    ax.set_title("PE-array utilisation: single- vs double-buffered weights", pad=14)
    ax.set_xticks(ticks, tick_labels)
    ax.set_xlim(-1, 3)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.tight_layout()
    save(fig, "fig4_double_buffering")


if __name__ == "__main__":
    fig_area_energy()
    fig_accuracy_energy()
    fig_size_sweep()
    fig_double_buffering()
