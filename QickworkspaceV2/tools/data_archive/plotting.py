"""Stable plot-id registry and hardware-independent archive plots."""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


class PlotRegistry:
    def __init__(self):
        self._plotters: dict[str, Callable] = {}

    def register(self, plot_id: str, plotter: Callable | None = None):
        def decorator(func):
            self._plotters[str(plot_id)] = func
            return func

        return decorator(plotter) if plotter is not None else decorator

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._plotters))

    def plot(self, record, *, kind=None, **kwargs):
        info = record.inspect()
        plot_id = str(kind or info.get("plot_id") or "generic_iq")
        plotter = self._plotters.get(plot_id, self._plotters["generic_iq"])
        return plotter(record.reader(), title=kwargs.pop("title", record.experiment_type), **kwargs)


default_plot_registry = PlotRegistry()


def _axis_for(array):
    if array.dims:
        name = array.dims[-1]
        values = array.axes.get(name, np.arange(array.shape[-1]))
        return name, values
    return "index", np.arange(array.shape[-1] if array.shape else 1)


@default_plot_registry.register("generic_iq")
@default_plot_registry.register("iq_fit_1d")
@default_plot_registry.register("rb_decay")
def plot_iq_1d(reader, *, title="Experiment", **_):
    raw = reader.raw("iq")
    iq = np.asarray(raw.values).squeeze()
    if iq.ndim != 1:
        return plot_nd(reader, title=title)
    axis_name, x = _axis_for(raw)
    fig, (ax_iq, ax_mag) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax_iq.plot(x, np.real(iq), label="I")
    ax_iq.plot(x, np.imag(iq), label="Q")
    ax_iq.set(xlabel=axis_name, ylabel="IQ", title="Raw IQ")
    ax_iq.legend()
    ax_mag.plot(x, np.abs(iq), label="|IQ|")
    if "fit_curve" in reader.analysis_keys():
        fit = np.asarray(reader.analysis("fit_curve").values).squeeze()
        if fit.shape == np.shape(x):
            ax_mag.plot(x, fit, "--", label="fit")
    ax_mag.set(xlabel=axis_name, ylabel="Amplitude", title="Magnitude and fit")
    ax_mag.legend()
    fig.suptitle(title)
    return fig


@default_plot_registry.register("single_shot_iq")
def plot_single_shot(reader, *, title="Single shot", **_):
    raw = reader.raw("iq")
    iq = np.asarray(raw.values)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    if iq.ndim < 2:
        iq = iq.reshape(1, -1)
    labels = raw.axes.get(raw.dims[0], np.arange(iq.shape[0])) if raw.dims else np.arange(iq.shape[0])
    for index, values in enumerate(iq):
        ax.scatter(np.real(values).reshape(-1), np.imag(values).reshape(-1), s=8, alpha=.45, label=str(labels[index]))
    ax.set(xlabel="I", ylabel="Q", title=title)
    ax.axis("equal")
    ax.legend(title="state")
    return fig


@default_plot_registry.register("density_matrix")
def plot_density_matrix(reader, *, title="Tomography", **_):
    key = "density_matrix"
    matrix = np.asarray(reader.analysis(key).values).squeeze()
    while matrix.ndim > 2:
        matrix = matrix[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, values, label in zip(axes, (matrix.real, matrix.imag), ("Real", "Imag")):
        image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(label)
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle(title)
    return fig


@default_plot_registry.register("ssh_optimization")
@default_plot_registry.register("sweep_2d")
def plot_nd(reader, *, title="Sweep", dataset="iq", **_):
    raw = reader.raw(dataset)
    values = np.abs(np.asarray(raw.values))
    while values.ndim > 2:
        values = np.nanmean(values, axis=-1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    if values.ndim == 2:
        image = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(image, ax=ax, label="|IQ|")
    else:
        ax.plot(values.reshape(-1))
    ax.set_title(title)
    return fig
