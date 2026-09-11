"""allxy: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
import numpy as np
from QickworkspaceV2.data.models import FitResult
from matplotlib.figure import Figure
from .program import ALLXY_IDEAL


def analyze_allxy(result):
    fits = {}
    for q, trace in result.traces.items():
        y = trace.signal(result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0))
        ground, excited = float(np.mean(y[:5])), float(np.mean(y[-4:]))
        if abs(excited - ground) < 1e-10:
            fits[q] = FitResult("allxy", False, message="No state contrast")
            continue
        population = (y - ground) / (excited - ground)
        error = float(np.sqrt(np.mean((population - np.asarray(ALLXY_IDEAL)) ** 2)))
        fits[q] = FitResult(
            "allxy",
            error < 0.1,
            {"normalized_rms_error": error},
            message="Endpoint-normalized AllXY; not a gate fidelity estimate",
        )
    return fits


def plot(result, **options):

    figure = Figure(figsize=(6 * len(result.targets), 4), layout="constrained")
    for ax, (q, trace) in zip(
        figure.subplots(1, len(result.targets), squeeze=False).flat, result.traces.items()
    ):
        y = trace.signal(result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0))
        contrast = np.mean(y[-4:]) - np.mean(y[:5])
        if abs(contrast) > 1e-12:
            ax.plot((y - np.mean(y[:5])) / contrast, "o", label="measured")
            ax.plot(ALLXY_IDEAL, label="ideal")
        else:
            ax.plot(y, "o", label="No reference contrast")
        ax.set(title=q, xlabel="AllXY sequence", ylabel="Normalized response")
        ax.legend()
    return figure


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="This diagnostic reports metrics without automatic calibration updates")


__all__ = ["analyze_allxy", "plot", "updates"]
