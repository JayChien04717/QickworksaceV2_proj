"""time_of_flight: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result
import numpy as np
from QickworkspaceV2.data.models import FitResult


def analyze(result):
    """Report envelope peak and an optional absolute-amplitude threshold crossing."""
    threshold = result.metadata.get("parameters", {}).get("tof_threshold")
    fits = {}
    for q, trace in result.traces.items():
        time = trace.coords["time"]
        amplitude = np.abs(trace.iq[:, 0])
        if not len(time) or not np.isfinite(amplitude).all():
            fits[q] = FitResult("tof_diagnostic", False, message="No finite TOF envelope")
            continue
        peak = int(np.argmax(amplitude))
        values = {"peak_amplitude": float(amplitude[peak]), "peak_time_us": float(time[peak])}
        units = {"peak_amplitude": "ADC", "peak_time_us": "us"}
        message = "Envelope diagnostic only; inspect the trace before adjusting trigger timing"
        if threshold is not None:
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("tof_threshold must be finite and nonnegative")
            crossings = np.flatnonzero(amplitude >= threshold)
            values["tof_threshold"] = float(threshold)
            units["tof_threshold"] = "ADC"
            if len(crossings):
                values["first_crossing_us"] = float(time[crossings[0]])
                units["first_crossing_us"] = "us"
            else:
                message += "; no threshold crossing"
        fits[q] = FitResult("tof_diagnostic", False, values, units=units, message=message)
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(
        reason="Inspect the timing trace and explicitly choose trigger and integration times"
    )


__all__ = ["analyze", "plot", "updates"]
