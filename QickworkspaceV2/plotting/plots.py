"""Matplotlib figures are returned to the caller; plotting never acquires or fits."""

from __future__ import annotations
import numpy as np


def plot_result(result, *, signal=None, event=0, residuals=True):
    from matplotlib.figure import Figure

    signal = signal or result.metadata.get("iq_process", "abs")
    targets = result.targets
    if not targets:
        raise ValueError("No traces to plot")
    figure = Figure(figsize=(6 * min(len(targets), 3), 4.7 * ((len(targets) + 2) // 3)), layout="constrained")
    columns = min(len(targets), 3)
    axes = figure.subplots((len(targets) + columns - 1) // columns, columns, squeeze=False)
    for ax, q in zip(axes.flat, targets):
        trace, fit = result[q], result.fits.get(q)
        dims = [d for d in trace.dims if d != "readout"]
        if trace.shots is not None and trace.shot_dims == ("shot", "readout") and not dims:
            for index in range(trace.shots.shape[-1]):
                iq = trace.shots[:, index]
                ax.scatter(iq.real, iq.imag, s=3, alpha=0.25, label=str(trace.coords["readout"][index]))
            ax.set(xlabel="I (ADC)", ylabel="Q (ADC)")
        else:
            values = trace.signal(signal, rotation_deg=trace.metadata.get("rotation_deg", 0))
            if dims:
                values = np.take(values, event, axis=trace.dims.index("readout"))
            if len(dims) == 1:
                dim = dims[0]
                x = trace.coords[dim]
                ax.plot(x, values, ".", ms=4, label="measurement")
                if fit and fit.x_fit:
                    ax.plot(
                        fit.x_fit,
                        fit.y_fit,
                        "-",
                        lw=1.5,
                        label=f"{fit.model} ({'pass' if fit.success else 'review'})",
                    )
                    if residuals:
                        order = np.argsort(x)
                        ax.plot(np.asarray(x)[order], fit.residuals, lw=0.7, alpha=0.6, label="residual")
                ax.set(xlabel=f"{dim} ({trace.units.get(dim, '')})", ylabel=f"IQ {signal}")
            elif len(dims) == 2:
                y, x = dims
                artist = ax.pcolormesh(trace.coords[x], trace.coords[y], values, shading="auto")
                figure.colorbar(artist, ax=ax, label=f"IQ {signal}")
                ax.set(xlabel=f"{x} ({trace.units.get(x, '')})", ylabel=f"{y} ({trace.units.get(y, '')})")
            elif not dims:
                ax.bar(np.arange(len(values.reshape(-1))), values.reshape(-1))
                ax.set(xlabel="readout", ylabel=f"IQ {signal}")
            else:
                raise ValueError("Select at most two sweep dimensions before plotting")
        ax.set_title(f"{q} · {result.experiment}")
        ax.grid(alpha=0.15)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(targets) :]:
        ax.set_visible(False)
    return figure


class LivePlot:
    """Progress subscriber. Pass to Session.run(on_progress=LivePlot())."""

    def __init__(self, **plot_options):
        self.plot_options, self.handle = plot_options, None

    def __call__(self, event):
        if event.get("result") is None:
            return
        from IPython.display import display

        figure = plot_result(event["result"], **self.plot_options)
        if self.handle is None:
            self.handle = display(figure, display_id=True)
        else:
            self.handle.update(figure)
