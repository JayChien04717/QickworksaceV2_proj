"""qubit_spec_ef: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces
from labtools.fitting import fit_lorentzian


def analyze(result):
    fits = analyze_traces(result, fit_lorentzian, signal=result.metadata["iq_process"])
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    fit = accepted_fit(result, target)
    return CalibrationUpdates(
        {"qb_freq_ef": fit["center"]},
        {"qb_freq_ef": f"qubits.{target}.transitions.ef.frequency_mhz"},
    )


__all__ = ["analyze", "plot", "updates"]
