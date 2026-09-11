"""power_rabi_chevron_ge: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces
from labtools.fitting import fit_rabi


def analyze(result):
    fits = analyze_traces(result, fit_rabi, signal=result.metadata["iq_process"])
    for fit in fits.values():
        for key in ("pi_gain", "pi2_gain"):
            if key in fit.parameters:
                fit.parameters[key] *= result.metadata["parameters"]["iterations"]
                if fit.errors.get(key) is not None:
                    fit.errors[key] *= result.metadata["parameters"]["iterations"]
        fit.message += "; repeated-pulse estimate, verify the single-pulse Rabi calibration"
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(
        reason="Repeated-pulse oscillation requires a single-pulse Rabi verification before updating gains"
    )


__all__ = ["analyze", "plot", "updates"]
