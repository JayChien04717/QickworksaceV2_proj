"""conditional_ramsey: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis import analyze_traces, fit_ramsey
from QickworkspaceV2.data.models import FitResult


def analyze_conditional(result):
    fits_g = analyze_traces(result, fit_ramsey, signal=result.metadata["iq_process"], event=0)
    fits_e = analyze_traces(result, fit_ramsey, signal=result.metadata["iq_process"], event=1)
    target = result.metadata["targets"][1]
    g, e = fits_g[target], fits_e[target]
    params, errors = {}, {}
    if "frequency" in g.parameters and "frequency" in e.parameters:
        params = {
            "frequency_g_mhz": g.parameters["frequency"],
            "frequency_e_mhz": e.parameters["frequency"],
            "frequency_difference_mhz": e.parameters["frequency"] - g.parameters["frequency"],
        }
        errors["frequency_difference_mhz"] = (
            (g.errors["frequency"] or 0) ** 2 + (e.errors["frequency"] or 0) ** 2
        ) ** 0.5
    result.metadata["conditional_fits"] = {"ground": g, "excited": e}
    return {
        target: FitResult(
            "conditional_ramsey",
            g.success and e.success,
            params,
            errors,
            {k: "MHz" for k in params},
            message="Difference of positive oscillation frequencies; signed ZZ requires detuning-branch verification",
        )
    }


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="This diagnostic reports metrics without automatic calibration updates")


__all__ = ["analyze_conditional", "plot", "updates"]
