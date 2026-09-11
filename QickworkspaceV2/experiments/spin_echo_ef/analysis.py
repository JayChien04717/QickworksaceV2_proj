"""spin_echo_ef: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces
from labtools.fitting import fit_exponential


def analyze(result):
    fits = analyze_traces(result, fit_exponential, signal=result.metadata["iq_process"])
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    fit = accepted_fit(result, target)
    return CalibrationUpdates(
        {"echo_ef_us": fit["tau"]},
        reason="Coherence times are working metrics, outside the strict device calibration schema",
    )


__all__ = ["analyze", "plot", "updates"]
