"""Shared, opt-in Matplotlib styling for QickworkspaceV2 figures."""

from __future__ import annotations

from cycler import cycler


COLORS = {
    "ink": "#172033",
    "muted": "#667085",
    "grid": "#D9E0EA",
    "panel": "#F7F9FC",
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "orange": "#EA580C",
    "green": "#15803D",
    "red": "#DC2626",
    "violet": "#7C3AED",
}

COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["violet"],
    COLORS["cyan"],
    COLORS["red"],
]


def style_axes(ax, *, grid: bool = True, panel: bool = False) -> None:
    """Apply the project figure style to one Axes without changing global rcParams."""
    ax.set_facecolor(COLORS["panel"] if panel else "white")
    ax.set_prop_cycle(cycler(color=COLOR_CYCLE))
    ax.tick_params(
        axis="both",
        colors=COLORS["muted"],
        labelsize=9,
        length=3,
        width=0.8,
    )
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    ax.title.set_color(COLORS["ink"])
    ax.title.set_fontweight("semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["grid"])
        ax.spines[side].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.grid(grid, color=COLORS["grid"], linewidth=0.75, alpha=0.65)


def style_figure(fig) -> None:
    """Set the neutral canvas used by project figures."""
    fig.patch.set_facecolor("white")


__all__ = ["COLORS", "COLOR_CYCLE", "style_axes", "style_figure"]
