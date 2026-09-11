"""Matplotlib figures are returned to the caller; plotting never acquires or fits."""

from __future__ import annotations
from io import BytesIO
from math import isfinite
from time import monotonic
import numpy as np


def plot_result(result, *, signal=None, event=0, residuals=True):
    from matplotlib.figure import Figure

    requested_signal = signal
    targets = result.targets
    if not targets:
        raise ValueError("No traces to plot")
    figure = Figure(figsize=(6 * min(len(targets), 3), 4.7 * ((len(targets) + 2) // 3)), layout="constrained")
    columns = min(len(targets), 3)
    axes = figure.subplots((len(targets) + columns - 1) // columns, columns, squeeze=False)
    for ax, q in zip(axes.flat, targets):
        trace, fit = result[q], result.fits.get(q)
        signal = (
            requested_signal or trace.metadata.get("fit_signal") or result.metadata.get("iq_process", "abs")
        )
        dims = [d for d in trace.dims if d != "readout"]
        if trace.shots is not None and trace.shot_dims == ("shot", "readout") and not dims:
            for index in range(trace.shots.shape[-1]):
                iq = trace.shots[:, index]
                ax.scatter(iq.real, iq.imag, s=3, alpha=0.25, label=str(trace.coords["readout"][index]))
            ax.set(xlabel="I (ADC)", ylabel="Q (ADC)")
            if fit and {"threshold", "rotation_deg"} <= fit.parameters.keys():
                angle = np.deg2rad(fit.parameters["rotation_deg"])
                normal = np.array([np.cos(angle), np.sin(angle)])
                center = fit.parameters["threshold"] * normal
                tangent = np.array([-normal[1], normal[0]])
                ax.axline(
                    center,
                    center + tangent,
                    color="black",
                    ls="--",
                    lw=1,
                    label=f"threshold; fidelity={fit.parameters.get('fidelity', float('nan')):.3f}",
                )
        else:
            values = trace.signal(signal, rotation_deg=trace.metadata.get("rotation_deg", 0))
            if signal == "population" and trace.metadata.get("fit_population_invert"):
                values = 1 - values
            if dims:
                values = np.take(values, event, axis=trace.dims.index("readout"))
            if len(dims) == 1:
                dim = dims[0]
                x = trace.coords[dim]
                ax.plot(x, values, ".", ms=4, label="measurement")
                if (
                    fit
                    and fit.x_fit
                    and event == trace.metadata.get("fit_event", 0)
                    and signal == trace.metadata.get("fit_signal", result.metadata.get("iq_process", "abs"))
                ):
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
    """Notebook PNG updates, independent of the active Matplotlib backend.

    Pass to Measurement.run or Session.run as on_progress=LivePlot().
    min_interval limits redraws in seconds; the first acquired result always renders; terminal events clear the display.
    It does not skip acquisition rounds or partial-data checkpoints.
    """

    def __init__(self, *, min_interval=0.1, progress=True, **plot_options):
        self.min_interval = float(min_interval)
        if not isfinite(self.min_interval) or self.min_interval < 0:
            raise ValueError("min_interval must be finite and nonnegative")
        self.plot_options, self.handle = plot_options, None
        self._last_update, self._run_id = None, None
        self._figure, self._canvas, self._signature = None, None, None
        self.progress = progress
        self._bar = None

    def _update_progress(self, event):
        completed, total = event.get("completed"), event.get("total")
        if not self.progress or completed is None or total is None:
            return
        if self._bar is None:
            from tqdm.auto import tqdm
            self._bar = tqdm(total=total, desc="Acquiring", unit="round", leave=True)
        elif completed < self._bar.n or total != self._bar.total:
            self._bar.reset(total=total)
        self._bar.update(completed - self._bar.n)

    def _update_figure(self, result):
        """Reuse artists for repeated rounds; rebuild when plot structure changes."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        event = self.plot_options.get("event", 0)
        requested_signal = self.plot_options.get("signal")
        signature = []
        for target, trace in result.traces.items():
            dims = tuple(d for d in trace.dims if d != "readout")
            signal = (
                requested_signal
                or trace.metadata.get("fit_signal")
                or result.metadata.get("iq_process", "abs")
            )
            signature.append(
                (
                    target,
                    trace.dims,
                    tuple(trace.iq.shape),
                    trace.shot_dims,
                    tuple(str(x) for x in trace.coords["readout"]),
                    signal,
                )
            )
        signature = (result.run_id, tuple(signature))
        if self._figure is None or signature != self._signature:
            if self._figure is not None:
                self._figure.clear()
            self._figure = plot_result(result, **self.plot_options)
            self._canvas = FigureCanvasAgg(self._figure)
            # Resolve layout once. Repeated constrained-layout solves dominated
            # small trace updates; the existing axes positions can be reused.
            self._figure.suptitle("Acquiring")
            self._canvas.draw()
            self._figure.set_layout_engine(None)
            self._signature = signature
        else:
            for ax, (target, trace) in zip(self._figure.axes, result.traces.items()):
                dims = [d for d in trace.dims if d != "readout"]
                signal = (
                    requested_signal
                    or trace.metadata.get("fit_signal")
                    or result.metadata.get("iq_process", "abs")
                )
                if trace.shots is not None and trace.shot_dims == ("shot", "readout") and not dims:
                    for collection, samples in zip(ax.collections, trace.shots.T):
                        collection.set_offsets(np.column_stack((samples.real, samples.imag)))
                    ax.dataLim.set_points(
                        np.array(
                            [
                                [trace.shots.real.min(), trace.shots.imag.min()],
                                [trace.shots.real.max(), trace.shots.imag.max()],
                            ]
                        )
                    )
                    ax.autoscale_view()
                else:
                    values = trace.signal(signal, rotation_deg=trace.metadata.get("rotation_deg", 0))
                    if signal == "population" and trace.metadata.get("fit_population_invert"):
                        values = 1 - values
                    if dims:
                        values = np.take(values, event, axis=trace.dims.index("readout"))
                    if len(dims) == 1:
                        ax.lines[0].set_data(trace.coords[dims[0]], values)
                        ax.relim()
                        ax.autoscale_view()
                    elif len(dims) == 2:
                        mesh = ax.collections[0]
                        mesh.set_array(values)
                        finite = values[np.isfinite(values)]
                        if finite.size:
                            mesh.set_clim(float(finite.min()), float(finite.max()))
                    else:
                        for bar, value in zip(ax.patches, values.reshape(-1)):
                            bar.set_height(value)
                        ax.relim()
                        ax.autoscale_view()
        return self._figure

    def __call__(self, event):
        if event.get("state") in {"completed", "cancelled", "failed"}:
            self.close()
            return
        self._update_progress(event)
        result = event.get("result")
        if result is None:
            return
        if (
            self._last_update is not None
            and result.run_id == self._run_id
            and monotonic() - self._last_update < self.min_interval
        ):
            return
        from IPython.display import Image, display

        # Start-to-start pacing includes rendering time, rather than adding the
        # requested interval after a costly render has already finished.
        started = monotonic()
        figure = self._update_figure(result)
        if event.get("completed") is not None and event.get("total") is not None:
            figure.suptitle(f"Acquiring: {event['completed']} / {event['total']}")
        # Bare Figure objects become text/plain when no inline formatter is registered.
        # Explicit PNG also avoids opening GUI windows in Qt-backed notebook kernels.
        with BytesIO() as buffer:
            self._canvas.print_png(buffer, pil_kwargs={"compress_level": 1})
            rendered = Image(data=buffer.getvalue(), format="png")
        if self.handle is None:
            self.handle = display(rendered, display_id=True)
        else:
            self.handle.update(rendered)
        self._run_id, self._last_update = result.run_id, started

    def close(self):
        """Remove only this live display, leaving logs and final figures intact."""
        from IPython.display import HTML

        if self._bar is not None:
            self._bar.close()
            self._bar = None

        if self.handle is not None:
            self.handle.update(HTML(""))
        if self._figure is not None:
            self._figure.clear()
        self.handle = self._figure = self._canvas = self._signature = None
        self._run_id = self._last_update = None
