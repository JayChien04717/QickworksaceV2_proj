"""This module provides setup-independent functions for evaluating electrical length vs flux bias."""

import numpy as np
import xarray as xr
from typing import Union

import matplotlib
import matplotlib.pyplot as plt


def plot_IQ(
    s21: xr.DataArray,
    f_min: float = None,
    f_max: float = None,
    flux_key="ifbl",
    freq_key="frequency",
    max_flux_coords=np.inf,
    ax=None,
):
    """Plot complex s21 data in the IQ plane. Only plots points within the specified frequency range.

    Parameters
    ----------
    s21 : xr.DataArray
        Value for ``s21``.
    f_min : float, default: None
        Value for ``f_min``.
    f_max : float, default: None
        Value for ``f_max``.
    flux_key : Any, default: 'ifbl'
        Value for ``flux_key``.
    freq_key : Any, default: 'frequency'
        Value for ``freq_key``.
    max_flux_coords : Any, default: np.inf
        Value for ``max_flux_coords``.
    ax : Any, default: None
        Matplotlib axes on which to draw.

    Returns
    -------
    Any
        Result of the operation.
    """
    if f_min is not None:
        s21 = s21.where((s21[freq_key] >= f_min), drop=True)
    if f_max is not None:
        s21 = s21.where((s21[freq_key] <= f_max), drop=True)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlabel("Re[S21]")
    ax.set_ylabel("Im[S21]")

    # Set the same scale for both real and imaginary axis, with origin in the center.
    scale = 1.05 * np.abs(s21).max()
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)

    flux_coords_to_plot = np.unique(s21[flux_key])
    while len(flux_coords_to_plot) > max_flux_coords:
        flux_coords_to_plot = flux_coords_to_plot[::2]  # Skip every other curve

    ax.plot([0.0], [0.0], "x", color="black", markersize=15)

    for flux, s21_at_one_flux in list(s21.groupby(flux_key)):
        if flux not in flux_coords_to_plot:
            continue
        ax.plot(
            s21_at_one_flux.real.to_numpy().transpose(),
            s21_at_one_flux.imag.to_numpy().transpose(),
            marker=".",
            linestyle="solid",
            linewidth=0.5,
        )

    return ax


def estimate_electrical_length(
    f: np.ndarray, s21: np.ndarray, f_min: float, f_max: float
):
    """Estimate electrical length based on one complex transmission vs frequency trace.
               Limit analysis to specified frequency range.

    Parameters
    ----------
    f : np.ndarray
        Value for ``f``.
    s21 : np.ndarray
        Value for ``s21``.
    f_min : float
        Value for ``f_min``.
    f_max : float
        Value for ``f_max``.

    Returns
    -------
    Any
        Result of the operation.
    """
    m = f >= f_min
    m *= f <= f_max
    assert all(np.diff(f[m])) >= 0, "The datapoints must be ordered in frequency."

    phase = np.unwrap(np.angle(s21[m]))
    gradient, offset = np.polyfit(f[m], phase, 1)
    return gradient / (
        2 * np.pi
    )  # Gradient is in radians/Hz. Convert to cycles/Hz = cycle * second


def plot_electrical_length(
    s21: xr.DataArray,
    f_min: float = 0,
    f_max: float = np.inf,
    flux_key="ifbl",
    freq_key="frequency",
    ax: plt.Axes = None,
):
    """Given complex S21 data, estimate and plot electrical length vs flux bias.

    Parameters
    ----------
    s21 : xr.DataArray
        Value for ``s21``.
    f_min : float, default: 0
        Value for ``f_min``.
    f_max : float, default: np.inf
        Value for ``f_max``.
    flux_key : Any, default: 'ifbl'
        Value for ``flux_key``.
    freq_key : Any, default: 'frequency'
        Value for ``freq_key``.
    ax : plt.Axes, default: None
        Matplotlib axes on which to draw.

    Returns
    -------
    Any
        Result of the operation.
    """
    if ax is None:
        fig, ax = plt.subplots()

    flux_values = []
    electrical_length = []
    for flux, s21_at_one_flux in list(s21.groupby(flux_key)):
        flux_values.append(flux)
        electrical_length.append(
            estimate_electrical_length(
                f=s21_at_one_flux[freq_key].to_numpy().reshape((-1,)),
                s21=s21_at_one_flux.to_numpy().reshape((-1,)),
                f_min=f_min,
                f_max=f_max,
            )
        )

    flux_values = np.array(flux_values)
    electrical_length = np.array(electrical_length)
    sorted_order = np.argsort(flux_values)
    flux_values = flux_values[sorted_order]
    electrical_length = electrical_length[sorted_order]

    electrical_length /= 1e-9  # Convert to nanoseconds
    print(
        f"Peak-to-peak electrical length modulation = {(electrical_length.max() - electrical_length.min()):.3f} ns"
    )

    ax.plot(
        [flux_values.min(), flux_values.max()],
        [electrical_length.min(), electrical_length.min()],
        color="gray",
        ls=":",
    )
    ax.plot(
        [flux_values.min(), flux_values.max()],
        [electrical_length.max(), electrical_length.max()],
        color="gray",
        ls=":",
    )
    ax.plot(flux_values, np.array(electrical_length))
    ax.set_xlabel(flux_key)
    ax.set_ylabel("Electrical length (ns)")

    return ax


def set_units_on_plot_axis(
    axis: Union[matplotlib.axis.XAxis, matplotlib.axis.YAxis],
    unit: float,
    unit_name=None,
    decimals: int = 0,
):
    """Set units on plot axis.

    Parameters
    ----------
    axis : Union[matplotlib.axis.XAxis, matplotlib.axis.YAxis]
        Value for ``axis``.
    unit : float
        Value for ``unit``.
    unit_name : Any, default: None
        Name of the unit.
    decimals : int, default: 0
        Value for ``decimals``.
    """
    axis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, pos: (f"%.{decimals:d}f") % (x / unit)
        )
    )
    lbl = axis.get_label()
    if unit_name is not None:
        lbl.set_text(f"{lbl.get_text()} ({unit_name})")
