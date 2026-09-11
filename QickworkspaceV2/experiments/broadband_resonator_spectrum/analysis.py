"""Broadband-only multi-resonator analysis, plots and update policy."""

from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure
from QickworkspaceV2.calibration.updates import CalibrationUpdates
from QickworkspaceV2.plotting.plots import plot_result
from labtools.fitting.fit_n_res import fit_n_resonators, _validated_trace
from QickworkspaceV2.data.models import FitResult
from .parameters import BroadbandParameters


def _settings(result):
    parameters = result.metadata.get("parameters", {})
    return BroadbandParameters.model_validate(
        {key: parameters[key] for key in ("count", "detection_options", "y_mode") if key in parameters}
    )


def analyze(result):
    """Run the original multi-resonator detector on complex IQ, with explicit MHz/Hz conversion."""
    settings = _settings(result)
    count = settings.count
    detection_options = settings.detection_options.model_dump()
    fits = {}
    for target, trace in result.traces.items():
        trace.metadata.pop("multi_resonator_fit", None)
        try:
            if trace.dims != ("frequency", "readout") or trace.iq.shape[1] != 1:
                raise ValueError("Broadband needs one frequency sweep and readout event")
            magnitude = abs(trace.iq[:, 0])
            if len(magnitude) and np.ptp(magnitude) <= 64 * np.finfo(float).eps * max(
                float(magnitude.max()), 1.0
            ):
                raise ValueError("No magnitude contrast; broadband candidates would be numerical artifacts")
            frequencies, details = fit_n_resonators(
                trace.coords["frequency"] * 1e6, trace.iq[:, 0], count=count, **detection_options
            )
            frequencies_mhz = frequencies / 1e6
            trace.metadata["multi_resonator_fit"] = {
                "count": count,
                "frequencies_mhz": frequencies_mhz.tolist(),
                "sample_freqs_mhz": (details["sample_freqs_hz"] / 1e6).tolist(),
                "details": details,
            }
            parameters = {f"resonance_{i}_mhz": float(value) for i, value in enumerate(frequencies_mhz, 1)}
            fits[target] = FitResult(
                "broadband",
                True,
                parameters,
                units={key: "MHz" for key in parameters},
                message=f"Located {count} resonators using phase-referenced dip detection and quadratic interpolation; verify a selected resonance in a narrow sweep.",
            )
        except (ValueError, RuntimeError) as exc:
            fits[target] = FitResult("broadband", False, message=str(exc))
    return fits


def plot(result, **options):
    """Return the original broadband frequency-marker plot."""
    settings = _settings(result)
    if any(
        trace.dims != ("frequency", "readout")
        or len(trace.iq) < 7
        or not np.isfinite(trace.iq).all()
        or not np.all(np.diff(trace.coords["frequency"]) > 0)
        for trace in result.traces.values()
    ):
        return plot_result(result, **options)
    figure = Figure(figsize=(9, 4.5 * len(result.targets)))
    for ax, (target, trace) in zip(
        figure.subplots(len(result.targets), 1, squeeze=False).flat, result.traces.items()
    ):
        frequencies_mhz = trace.metadata.get("multi_resonator_fit", {}).get("frequencies_mhz", [])
        plot_n_resonators(
            trace.coords["frequency"] * 1e6,
            trace.iq[:, 0],
            np.asarray(frequencies_mhz) * 1e6,
            y_mode=options.get("y_mode", options.get("signal") or settings.y_mode),
            ax=ax,
            marker_kwargs=options.get("marker_kwargs"),
        )
        ax.set_title(target + " / " + ax.get_title())
    return figure


def updates(result, target):
    """Broadband identifies resonators; selection requires a separate narrow sweep."""
    return CalibrationUpdates(
        reason="Select a broadband candidate and verify it with a narrow resonance sweep"
    )


def plot_n_resonators(
    freq_hz: np.ndarray,
    s21: np.ndarray,
    fitted_freqs_hz: np.ndarray,
    *,
    y_mode: str = "abs",
    ax=None,
    marker_kwargs: dict | None = None,
):
    """Plot an S21 trace and mark fitted resonator frequencies.

    Parameters are expressed in hertz to match :func:`fit_n_resonators`; the
    displayed x-axis is converted to megahertz for readability.

    Returns
    -------
    fig, ax, resonator_freqs_hz
        The Matplotlib figure and axes, followed by the sorted frequencies
        shown on the plot in hertz.
    """

    frequency, transmission = _validated_trace(freq_hz, s21)
    fitted = np.asarray(fitted_freqs_hz, dtype=float)
    if fitted.ndim != 1:
        raise ValueError("fitted_freqs_hz must be a 1D array.")
    if not np.all(np.isfinite(fitted)):
        raise ValueError("fitted_freqs_hz must contain only finite values.")

    y_mode = str(y_mode).lower()
    if y_mode == "abs":
        y_values = np.abs(transmission)
        ylabel = "|S21| (ADC Units)"
    elif y_mode == "db":
        floor = np.finfo(float).tiny
        y_values = 20 * np.log10(np.maximum(np.abs(transmission), floor))
        ylabel = "|S21| (dB)"
    elif y_mode == "phase":
        y_values = np.unwrap(np.angle(transmission))
        ylabel = "Phase (rad)"
    else:
        raise ValueError("y_mode must be one of 'abs', 'db', or 'phase'.")

    frequency_mhz = frequency / 1e6
    fitted = np.sort(fitted)
    fitted_mhz = fitted / 1e6
    if ax is None:
        fig = Figure(figsize=(8, 5))
        ax = fig.subplots()
    else:
        fig = ax.figure

    ax.plot(frequency_mhz, y_values, color="C0", label="S21")
    marker_style = {
        "color": "C3",
        "linestyle": "--",
        "linewidth": 1.2,
        "alpha": 0.8,
    }
    if marker_kwargs is not None:
        marker_style.update(marker_kwargs)

    for index, (resonator_hz, resonator_mhz) in enumerate(zip(fitted, fitted_mhz), start=1):
        ax.axvline(resonator_mhz, **marker_style)
        nearest = int(np.argmin(np.abs(frequency - resonator_hz)))
        ax.scatter(
            [resonator_mhz],
            [y_values[nearest]],
            color=marker_style.get("color", "C3"),
            s=28,
            zorder=3,
        )
        ax.annotate(
            f"R{index}\n{resonator_mhz:.4f} MHz",
            xy=(resonator_mhz, y_values[nearest]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8,
            color=marker_style.get("color", "C3"),
        )

    ax.set_title(f"Multi-Resonator One-Tone Fit ({fitted.size} resonators)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    return fig, ax, fitted


__all__ = ["analyze", "plot", "plot_n_resonators", "updates"]
