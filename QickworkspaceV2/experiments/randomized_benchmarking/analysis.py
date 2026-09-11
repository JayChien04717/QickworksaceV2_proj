"""randomized_benchmarking: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
import numpy as np
from QickworkspaceV2.analysis.fitting import fit_rb
from matplotlib.figure import Figure
from QickworkspaceV2.plotting.plots import plot_result


def analyze_rb(result):
    # Direct Notebook acquisition and catalog-built plans share the native cfg.
    repeated = result.metadata["resolved_config"]["rb_depths"]
    depths = np.asarray(list(dict.fromkeys(repeated)))
    repeats = repeated.count(repeated[0])
    if list(repeated) != list(np.repeat(depths, repeats)):
        raise ValueError("RB requires equal sequence counts grouped by depth")

    fits = {}
    for q, trace in result.traces.items():
        values = trace.signal(
            result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0)
        ).reshape(len(depths), repeats)
        y = values.mean(axis=1)
        settings = dict(result.metadata.get("parameters", {}).get("fit", {}))
        fit = fit_rb(depths, y, **settings)
        fit.message += "; unweighted sequence means; EPC assumes Markovian single-qubit Clifford noise"
        fits[q] = fit
    return fits


def plot(result, **options):
    if any(t.dims != ("readout",) for t in result.traces.values()):
        return plot_result(result, **options)
    repeated = result.metadata["resolved_config"]["rb_depths"]
    depths, count = list(dict.fromkeys(repeated)), repeated.count(repeated[0])
    figure = Figure(figsize=(6 * len(result.targets), 4), layout="constrained")
    for ax, (q, trace) in zip(
        figure.subplots(1, len(result.targets), squeeze=False).flat, result.traces.items()
    ):
        values = trace.signal(
            result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0)
        ).reshape(len(depths), count)
        ax.errorbar(depths, values.mean(axis=1), yerr=values.std(axis=1, ddof=1) / np.sqrt(count), fmt="o")
        fit = result.fits.get(q)
        if fit and fit.x_fit:
            ax.plot(fit.x_fit, fit.y_fit, label="RB fit" + (" (review)" if not fit.success else ""))
            ax.legend()
        ax.set(title=q, xlabel="Clifford depth", ylabel="Sequence response")
    return figure


def interleaved_error(reference, result, target):
    """Markovian interleaved RB estimate against an accepted reference."""
    from QickworkspaceV2.calibration.updates import accepted_fit

    p_ref = accepted_fit(reference, target)["decay_probability"]
    p_gate = accepted_fit(result, target)["decay_probability"]
    if p_ref <= 0:
        raise ValueError("RB reference decay probability must be positive")
    return (1 - p_gate / p_ref) / 2


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="RB reports gate-error metrics without changing pulse calibration")


__all__ = ["analyze_rb", "plot", "updates"]
