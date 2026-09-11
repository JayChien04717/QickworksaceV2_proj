"""qubit_temperature: analysis."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates

import numpy as np
from scipy.constants import h, k
from QickworkspaceV2.analysis import analyze_traces
from labtools.fitting import fit_rabi
from QickworkspaceV2.data.models import FitResult
from matplotlib.figure import Figure


def analyze(result):
    thermal = analyze_traces(result, fit_rabi, signal=result.metadata["iq_process"], event=0)
    reference = analyze_traces(result, fit_rabi, signal=result.metadata["iq_process"], event=1)
    fits = {}
    result.metadata["temperature_rabi_fits"] = {"thermal": thermal, "ge_prepared": reference}
    cfg = result.metadata["resolved_config"]
    for q in result.targets:
        first, second = thermal[q], reference[q]
        if not first.success or not second.success:
            fits[q] = FitResult("thermal_population", False, message="Both EF Rabi amplitude fits must pass")
            continue
        a, b = abs(first.parameters["amplitude"]), abs(second.parameters["amplitude"])
        pe = a / (a + b)
        if not 0 < pe < 0.5:
            fits[q] = FitResult(
                "thermal_population",
                False,
                {"excited_population": pe},
                message="Population lies outside the positive-temperature two-level regime",
            )
            continue
        frequency_hz = float(cfg["qubits"][q]["qb_freq_ge"]) * 1e6
        temperature = -h * frequency_hz / (k * np.log(pe / (1 - pe)))
        fits[q] = FitResult(
            "thermal_population",
            True,
            {"excited_population": pe, "effective_temperature_mk": temperature * 1000},
            units={"effective_temperature_mk": "mK"},
            message="Assumes negligible f population, equal EF response and ideal GE inversion; not a thermometer calibration",
        )
    return fits


def _fit_value(fit, key):
    return fit.get(key, []) if isinstance(fit, dict) else getattr(fit, key)


def plot(result, **options):
    figure = Figure(figsize=(10, 4 * len(result.targets)), layout="constrained")
    records = result.metadata.get("temperature_rabi_fits", {})
    for axes, (q, trace) in zip(
        figure.subplots(len(result.targets), 2, squeeze=False), result.traces.items()
    ):
        dim = next(d for d in trace.dims if d != "readout")
        y = trace.signal(result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0))
        for index, label in enumerate(("thermal", "ge_prepared")):
            ax = axes[index]
            ax.plot(trace.coords[dim], y[:, index], "o")
            fit = records.get(label, {}).get(q)
            if fit and _fit_value(fit, "x_fit"):
                ax.plot(_fit_value(fit, "x_fit"), _fit_value(fit, "y_fit"))
            ax.set(title=q + " / " + label, xlabel="EF gain", ylabel="IQ response")
    return figure


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="This diagnostic reports metrics without automatic calibration updates")


__all__ = ["analyze", "plot", "updates"]
