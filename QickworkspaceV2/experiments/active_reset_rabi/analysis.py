"""active_reset_rabi: analysis."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates

from QickworkspaceV2.analysis import fit_rabi
from QickworkspaceV2.data.models import FitResult
from matplotlib.figure import Figure
from QickworkspaceV2.plotting.plots import plot_result


def analyze(result):
    fits = {}
    options = result.metadata.get("parameters", {})
    for q, trace in result.traces.items():
        try:
            population = trace.population(
                trace.metadata["threshold"], rotation_deg=trace.metadata["rotation_deg"]
            )
            if options.get("reset_excited_if", ">=") == "<":
                population = 1 - population
            trace.metadata.update(
                fit_signal="population",
                fit_event=0,
                fit_population_invert=options.get("reset_excited_if", ">=") == "<",
            )
            axis = next(d for d in trace.dims if d != "readout")
            fit = fit_rabi(
                trace.coords[axis], population[:, 0], unit=trace.units[axis], **options.get("fit", {})
            )
            fit.parameters.update(
                pre_reset_mean=float(population[:, 0].mean()),
                post_reset_mean=float(population[:, 1].mean()),
                post_reset_max=float(population[:, 1].max()),
            )
            fit.message += "; reset effectiveness requires comparison with the never-reset control"
            fits[q] = fit
        except (ValueError, KeyError) as exc:
            fits[q] = FitResult("active_reset", False, message=str(exc))
    return fits


def plot(result, **options):
    if any(trace.dims != ("gain", "readout") for trace in result.traces.values()):
        return plot_result(result, **options)
    figure = Figure(figsize=(10, 4 * len(result.targets)), layout="constrained")
    for axes, (q, trace) in zip(
        figure.subplots(len(result.targets), 2, squeeze=False), result.traces.items()
    ):
        population = trace.population(
            trace.metadata["threshold"], rotation_deg=trace.metadata["rotation_deg"]
        )
        if trace.metadata.get("fit_population_invert"):
            population = 1 - population
        for index, label in enumerate(("Pre-reset", "Post-reset")):
            ax = axes[index]
            ax.plot(trace.coords["gain"], population[:, index], "o-")
            fit = result.fits.get(q)
            if index == 0 and fit and fit.x_fit:
                ax.plot(fit.x_fit, fit.y_fit)
            ax.set(title=q + " / " + label, xlabel="Rabi gain", ylabel="Excited population")
    return figure


def plot_comparison(runs, target):
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    for mode, result in runs.items():
        trace = result[target]
        population = trace.population(
            trace.metadata["threshold"], rotation_deg=trace.metadata["rotation_deg"]
        )
        if trace.metadata.get("fit_population_invert"):
            population = 1 - population
        for event, ax in enumerate(axes):
            ax.plot(trace.coords["gain"], population[:, event], "o-", label=mode)
    for ax, title in zip(axes, ["Pre-reset", "Post-reset"]):
        ax.set(title=title, xlabel="Rabi gain", ylabel="Excited population")
        ax.legend()
    return figure


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="Compare reset controls before choosing feedback settings")


__all__ = ["analyze", "plot", "updates"]
