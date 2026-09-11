"""ramsey_ge: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces
from labtools.fitting import fit_ramsey


def analyze(result):
    fits = analyze_traces(result, fit_ramsey, signal=result.metadata["iq_process"])
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates_pair(results, target):
    """Resolve correction sign with two measured zero-detuning Ramsey runs."""
    import math

    if not {"minus", "plus"} <= results.keys():
        raise ValueError("Both Ramsey offset acquisitions must complete")
    minus, plus = results["minus"], results["plus"]
    for result in (minus, plus):
        if result.experiment != "ramsey_ge" or result.targets != (target,):
            raise ValueError("Ramsey pair must measure the same selected GE target")
        if result.metadata["resolved_config"].get("detuning_mhz", 0) != 0:
            raise ValueError("Ramsey pair requires zero virtual detuning")
    f_minus = accepted_fit(minus, target)["frequency"]
    f_plus = accepted_fit(plus, target)["frequency"]
    drive_minus = minus.metadata["resolved_config"]["qubits"][target]["qb_freq_ge"]
    drive_plus = plus.metadata["resolved_config"]["qubits"][target]["qb_freq_ge"]
    center, offset = (drive_plus + drive_minus) / 2, (drive_plus - drive_minus) / 2
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError("Ramsey pair needs distinct, ordered drive frequencies")
    frequency = center + (f_minus**2 - f_plus**2) / (4 * offset)
    return CalibrationUpdates(
        {"qb_freq_ge": frequency},
        {"qb_freq_ge": f"qubits.{target}.transitions.ge.frequency_mhz"},
    )


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    fit = accepted_fit(result, target)
    return CalibrationUpdates(
        {"ramsey_ge_us": fit["tau"]},
        reason="Coherence times are working metrics, outside the strict device calibration schema",
    )


__all__ = ["analyze", "plot", "updates"]
