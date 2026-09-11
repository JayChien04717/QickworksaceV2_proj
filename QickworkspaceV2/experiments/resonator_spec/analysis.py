"""resonator_spec: single-resonator analysis, plot and calibration."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces, fit_lorentzian


def analyze(result):
    return analyze_traces(result, fit_lorentzian, signal=result.metadata["iq_process"])


def plot(result, **options):
    return plot_result(result, **options)


def updates(result, target):
    fit = accepted_fit(result, target)
    group = result.metadata["resolved_config"]["qubits"][target]["readout_group"]
    return CalibrationUpdates(
        {"res_freq_ge": fit["center"]},
        {"res_freq_ge": f"readout_groups.{group}.members.{target}.frequency_mhz"},
    )


__all__ = ["analyze", "plot", "updates"]
