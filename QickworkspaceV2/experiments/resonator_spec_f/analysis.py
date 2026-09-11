"""resonator_spec_f: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces, fit_lorentzian


def analyze(result):
    return analyze_traces(result, fit_lorentzian, signal=result.metadata["iq_process"])


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(
        reason="Second-excited-state resonance does not replace the ground-state readout frequency"
    )


__all__ = ["analyze", "plot", "updates"]
