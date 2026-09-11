"""time_rabi_ge: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces
from labtools.fitting import fit_rabi


def analyze(result):
    fits = analyze_traces(result, fit_rabi, signal=result.metadata["iq_process"])
    for fit in fits.values():
        for old, new in (("pi_gain", "pi_length_us"), ("pi2_gain", "pi2_length_us")):
            if old in fit.parameters:
                fit.parameters[new] = fit.parameters.pop(old)
                fit.errors[new] = fit.errors.pop(old)
                fit.units.pop(old, None)
                fit.units[new] = "us"
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(
        reason="Review pulse shape and pi/pi2 durations before editing the working pulse settings"
    )


__all__ = ["analyze", "plot", "updates"]
