"""Publication-friendly fit-result figures shared by Analysis classes."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .theme import COLORS, style_axes, style_figure

# ── Quality → colour mapping ──────────────────────────────────────────────────

_QUALITY_COLOR = {
    "good": COLORS["green"],
    "warning": COLORS["orange"],
    "bad": COLORS["red"],
    "no_information": COLORS["muted"],
}

# ── Style constants ───────────────────────────────────────────────────────────

_DATA_KW = dict(
    marker="o", ms=4.2, markeredgewidth=0, alpha=0.78,
    lw=1.15, color=COLORS["blue"],
)
_FIT_KW = dict(lw=2.35, color=COLORS["orange"], zorder=5)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_fit_result(
    xpts: np.ndarray,
    iq_data: np.ndarray,
    simfunc,
    fit_params,
    *,
    x_label: str = "x",
    title: str = "",
    result_text: str = "",
    quality: str = "no_information",
    extra_lines=None,
    fit_channel: str = "abs",
) -> plt.Figure:
    """Render a clean fit dashboard with one primary and three context panels.

    Parameters
    ----------
    xpts        : sweep axis (x-values).
    iq_data     : complex IQ array (raw_iq from ExperimentData).
    simfunc     : fit function  ``f(x, *params) → y``.
    fit_params  : pre-computed fit parameters (``None`` → skip fit overlay).
    x_label     : shared x-axis label.
    title       : figure suptitle.
    result_text : newline-separated key results shown in the main panel.
    quality     : ``"good" | "warning" | "bad" | "no_information"``.
    extra_lines : list of dicts passed to ``ax.axvline()`` in every panel,
                  e.g. ``[{"x": 0.5, "color": "r", "ls": "--", "label": "π"}]``.
    """
    channels = {
        "Amplitude": np.abs(iq_data),
        "Phase":     np.unwrap(np.angle(iq_data)),
        "I":         iq_data.real,
        "Q":         iq_data.imag,
    }
    channel_label = {
        "abs": "Amplitude",
        "amplitude": "Amplitude",
        "amp": "Amplitude",
        "real": "I",
        "i": "I",
        "imag": "Q",
        "q": "Q",
        "phase": "Phase",
    }.get(str(fit_channel).lower(), "Amplitude")

    x_fit = np.linspace(xpts[0], xpts[-1], 600) if fit_params is not None else None
    fit_y = simfunc(x_fit, *fit_params) if fit_params is not None else None

    quality_key = str(quality).lower()
    q_color = _QUALITY_COLOR.get(quality_key, COLORS["muted"])

    fig = plt.figure(figsize=(12.5, 6.3), layout="constrained")
    style_figure(fig)
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[2.35, 1],
        height_ratios=[1, 1, 1],
        hspace=0.16,
        wspace=0.12,
    )

    # The fitted channel gets the visual priority; the other channels provide
    # context without repeating the primary trace.
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.plot(
        xpts,
        channels[channel_label],
        label=f"{channel_label} data",
        **_DATA_KW,
    )

    if fit_y is not None:
        ax_main.plot(x_fit, fit_y, label="Best fit", **_FIT_KW)

    if extra_lines:
        for kw in extra_lines:
            ax_main.axvline(**kw)
    ax_main.set_xlabel(x_label, fontsize=10.5, labelpad=8)
    main_unit = "rad" if channel_label == "Phase" else "ADC unit"
    ax_main.set_ylabel(f"{channel_label} ({main_unit})", fontsize=10.5, labelpad=8)
    style_axes(ax_main)
    legend = ax_main.legend(
        loc="lower left",
        fontsize=8.5,
        ncols=2,
        frameon=True,
        borderpad=0.7,
        handlelength=2.2,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(COLORS["grid"])
    legend.get_frame().set_linewidth(0.8)

    if result_text:
        ax_main.text(
            0.97, 0.95, result_text,
            transform=ax_main.transAxes,
            fontsize=9.2, va="top", ha="right", family="monospace",
            color=COLORS["ink"],
            linespacing=1.45,
            bbox=dict(
                boxstyle="round,pad=0.65,rounding_size=0.18",
                facecolor=COLORS["panel"],
                edgecolor=COLORS["grid"],
                alpha=0.96,
                linewidth=0.9,
            ),
        )

    badge_label = quality_key.upper().replace("_", " ")
    ax_main.text(
        0.03, 0.95, f"●  {badge_label}",
        transform=ax_main.transAxes,
        fontsize=8.3,
        va="top",
        ha="left",
        color=q_color,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.14",
            facecolor="white",
            edgecolor=q_color,
            linewidth=0.8,
            alpha=0.95,
        ),
    )

    context_channels = [key for key in ("Amplitude", "I", "Q", "Phase")
                        if key != channel_label][:3]
    for row, key in enumerate(context_channels):
        ax = fig.add_subplot(gs[row, 1], sharex=ax_main)
        ax.plot(
            xpts,
            channels[key],
            lw=1.05,
            color=COLORS["cyan"],
            alpha=0.88,
        )
        if extra_lines:
            for kw in extra_lines:
                marker_kw = {k: v for k, v in kw.items() if k != "label"}
                marker_kw["alpha"] = min(float(marker_kw.get("alpha", 1.0)), 0.45)
                ax.axvline(**marker_kw)
        ax.set_title(key, loc="left", fontsize=9.5, pad=5)
        ax.set_ylabel("rad" if key == "Phase" else "ADC", fontsize=8.5)
        if row == len(context_channels) - 1:
            ax.set_xlabel(x_label, fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        style_axes(ax, panel=True)

    if title:
        fig.suptitle(
            title,
            fontsize=14,
            fontweight="semibold",
            color=COLORS["ink"],
            x=0.01,
            ha="left",
        )

    plt.show()
    return fig


__all__ = ["plot_fit_result"]
