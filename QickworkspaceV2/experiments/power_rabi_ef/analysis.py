"""power_rabi_ef: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces, fit_rabi


def analyze(result):
    fits = analyze_traces(result, fit_rabi, signal=result.metadata["iq_process"])
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    fit = accepted_fit(result, target)
    cfg = result.metadata["resolved_config"]["qubits"][target]
    return CalibrationUpdates(
        {
            "pi_gain_ef": fit["pi_gain"],
            "pi2_gain_ef": fit["pi2_gain"],
            "sigma_ef": cfg["sigma_ef"],
            "pulse_type_ef": cfg["pulse_type_ef"],
        },
        {
            "pi_gain_ef": f"qubits.{target}.transitions.ef.pulse.pi_gain",
            "pi2_gain_ef": f"qubits.{target}.transitions.ef.pulse.pi2_gain",
            "sigma_ef": f"qubits.{target}.transitions.ef.pulse.sigma_us",
        },
    )


__all__ = ["analyze", "plot", "updates"]
