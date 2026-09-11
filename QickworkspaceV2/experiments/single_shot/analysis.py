"""single_shot: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit
from dataclasses import replace
from QickworkspaceV2.analysis import analyze_single_shot
import numpy as np
from matplotlib.figure import Figure
from QickworkspaceV2.plotting.plots import plot_result


def analyze(result):
    cfg = result.metadata["resolved_config"]
    axis = cfg.get("classifier_axis", "auto")
    if axis not in ("auto", "I", "Q"):
        raise ValueError("classifier_axis must be auto, I or Q")
    fits = {}
    for q, trace in result.traces.items():
        qc = cfg["qubits"][q]
        rotation = qc["ro_phase"] if axis == "auto" else (0 if axis == "I" else 90)
        fits.update(
            analyze_single_shot(
                replace(result, traces={q: trace}),
                threshold=qc.get("ro_threshold"),
                rotation_deg=rotation,
                fixed_axis=axis != "auto",
            )
        )
    return fits


def plot(result, **options):
    if any(trace.dims != ("readout",) for trace in result.traces.values()):
        return plot_result(result, **options)
    figure = Figure(figsize=(10, 4 * len(result.targets)), layout="constrained")
    for axes, (q, trace) in zip(
        figure.subplots(len(result.targets), 2, squeeze=False), result.traces.items()
    ):
        ax, matrix_ax = axes
        if trace.shots is not None:
            for index, label in enumerate(trace.coords["readout"]):
                samples = trace.shots[:, index]
                ax.scatter(samples.real, samples.imag, s=3, alpha=0.3, label=str(label))
            ax.legend()
        fit = result.fits.get(q)
        if fit and "threshold" in fit.parameters:
            angle = np.deg2rad(fit.parameters["rotation_deg"])
            normal = np.array([np.cos(angle), np.sin(angle)])
            center = fit.parameters["threshold"] * normal
            ax.axline(center, center + [-normal[1], normal[0]], color="black", ls="--")
        ax.set(title=q, xlabel="I (ADC)", ylabel="Q (ADC)")
        confusion = trace.metadata.get("confusion_matrix")
        if confusion is not None:
            matrix = np.asarray(confusion)
            image = matrix_ax.imshow(matrix, vmin=0, vmax=1)
            for (row, col), value in np.ndenumerate(matrix):
                matrix_ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="red")
            figure.colorbar(image, ax=matrix_ax)
        matrix_ax.set(xlabel="Assigned state", ylabel="Prepared state", title="Assignment confusion")
    return figure


def plot_grid(rows, gains, lengths):
    """Plot the measured grid, leaving unmeasured/interrupted points blank."""
    fidelity = np.full((len(lengths), len(gains)), np.nan)
    for row in rows:
        i = list(lengths).index(row["length_us"])
        j = list(gains).index(row["gain"])
        fidelity[i, j] = row["fidelity"]
    figure = Figure(figsize=(6, 4), layout="constrained")
    ax = figure.subplots()
    image = ax.pcolormesh(gains, lengths, fidelity, shading="auto", vmin=0, vmax=1)
    figure.colorbar(image, ax=ax, label="Held-out fidelity")
    ax.set(xlabel="Readout gain", ylabel="Integration time (us)")
    return figure


def best_readout(rows):
    valid = [row for row in rows if row["passed"] and row["complete"] and np.isfinite(row["fidelity"])]
    if not valid:
        raise ValueError("No readout grid point passed")
    return max(valid, key=lambda row: row["fidelity"])["run_id"]


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    fit = accepted_fit(result, target)
    cfg = result.metadata["resolved_config"]["qubits"][target]
    root = f"readout_groups.{cfg['readout_group']}.members.{target}"
    return CalibrationUpdates(
        {
            "ro_threshold": fit["threshold"],
            "ro_phase": fit["rotation_deg"],
            "res_gain_ge": cfg["res_gain_ge"],
            "ro_length": cfg["ro_length"],
            "res_length": cfg["res_length"],
        },
        {
            "ro_threshold": root + ".threshold",
            "ro_phase": root + ".rotation_deg",
            "res_gain_ge": root + ".gain",
        },
    )


__all__ = ["analyze", "plot", "updates"]
