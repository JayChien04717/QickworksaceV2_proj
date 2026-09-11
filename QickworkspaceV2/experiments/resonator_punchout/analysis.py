"""Punch-out map summary, final heatmap and explicit update policy."""

from QickworkspaceV2.analysis.diagnostics import summarize_traces
from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result


def analyze(result):
    """Summarize the measured 2D IQ map, as in the original visual diagnostic."""
    return summarize_traces(result)


def plot(result, **options):
    """Show frequency horizontally and readout gain vertically."""
    return plot_result(result, **options)


def updates(result, target):
    """Readout power selection requires reviewing the map, not an automatic fit."""
    return CalibrationUpdates(reason="Inspect the punch-out map before selecting readout power")
