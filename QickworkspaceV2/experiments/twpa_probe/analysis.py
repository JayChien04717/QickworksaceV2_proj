"""twpa_probe: analysis."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates

from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.analysis.diagnostics import summarize_traces


analyze = summarize_traces


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="This diagnostic reports metrics without automatic calibration updates")


__all__ = ["analyze", "plot", "updates"]
