"""drag_ef: analysis."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit

from QickworkspaceV2.plotting.plots import plot_result
from QickworkspaceV2.data.models import FitResult


def analyze(result):
    fits = {}
    for q, trace in result.traces.items():
        g, e, returned = trace.iq
        contrast = abs(e - g)
        if contrast < 1e-10:
            fits[q] = FitResult("drag_return", False, message="No reference-state contrast")
            continue
        projection = float(((returned - g) * (e - g).conjugate()).real / contrast**2)
        fits[q] = FitResult(
            "drag_return",
            -0.1 <= projection <= 1.1,
            {"return_error_signal": projection},
            message="IQ projected on measured reference states; includes leakage and preparation errors",
        )
    return fits


def plot(result, **options):
    """Final figure for this experiment; customize labels/layout here."""
    return plot_result(result, **options)


def plot_scan(runs, target):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6, 4), layout="constrained")
    ax = figure.subplots()
    ax.plot(
        [row["alpha"] for row in runs],
        [row["result"].fits[target].parameters.get("return_error_signal", float("nan")) for row in runs],
        "o-",
    )
    ax.set(xlabel="DRAG alpha", ylabel="Return-error signal")
    return figure


def best_result(runs, target):
    import math

    valid = []
    for row in runs:
        result = row["result"]
        if result.experiment != "drag_ef" or result.targets != (target,):
            raise ValueError("DRAG scan must measure one selected ef target")
        try:
            error = accepted_fit(result, target)["return_error_signal"]
        except ValueError:
            continue
        if math.isfinite(error):
            valid.append((abs(error), result))
    if not valid:
        raise ValueError("No DRAG point passed")
    return min(valid, key=lambda item: item[0])[1]


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    accepted_fit(result, target)
    cfg = result.metadata["resolved_config"]["qubits"][target]
    return CalibrationUpdates(
        {
            "drag_alpha_ef": cfg["drag_alpha_ef"],
            "drag_delta_ef": cfg["drag_delta_ef"],
            "shape_ef": "drag",
            "pulse_type_ef": "arb",
        },
        reason="DRAG also changes pulse shape; review it in the working configuration",
    )


__all__ = ["analyze", "plot", "updates"]
