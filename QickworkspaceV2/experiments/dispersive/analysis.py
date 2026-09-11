"""dispersive: analysis."""

from QickworkspaceV2.calibration.updates import CalibrationUpdates, accepted_fit

import numpy as np
from QickworkspaceV2 import FitResult
from matplotlib.figure import Figure


def analyze(data):
    fits = {}
    for target, trace in data.traces.items():
        shots = trace.shots
        if shots is None or trace.shot_dims != ("frequency", "shot", "readout"):
            raise ValueError("Dispersive analysis requires a frequency axis and paired shots")
        delta = abs(shots[:, :, 1].mean(axis=1) - shots[:, :, 0].mean(axis=1))
        noise = np.sqrt((np.var(shots[:, :, 0], axis=1) + np.var(shots[:, :, 1], axis=1)) / 2)
        snr = np.divide(delta, noise, out=np.zeros_like(delta), where=noise > 0)
        index = int(np.argmax(snr))
        trace.metadata["readout_snr"] = snr
        fits[target] = FitResult(
            "readout_snr",
            bool(np.isfinite(snr).all() and snr[index] >= 2),
            {"best_frequency_mhz": float(trace.coords["frequency"][index]), "snr": float(snr[index])},
            message="Measured mean separation / pooled complex standard deviation",
        )
    return fits


def plot(result, **options):
    figure = Figure(figsize=(10, 4 * len(result.targets)), layout="constrained")
    for axes, (q, trace) in zip(
        figure.subplots(len(result.targets), 2, squeeze=False), result.traces.items()
    ):
        frequency = trace.coords["frequency"]
        for index, label in enumerate(trace.coords["readout"]):
            axes[0].plot(frequency, abs(trace.iq[:, index]), label=str(label))
        axes[0].legend()
        axes[0].set(title=q, xlabel="Readout frequency (MHz)", ylabel="IQ abs")
        axes[1].plot(frequency, trace.metadata.get("readout_snr", np.zeros(len(frequency))))
        axes[1].set(xlabel="Readout frequency (MHz)", ylabel="Measured SNR")
    return figure


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    fit = accepted_fit(result, target)
    group = result.metadata["resolved_config"]["qubits"][target]["readout_group"]
    return CalibrationUpdates(
        {"res_freq_ge": fit["best_frequency_mhz"]},
        {"res_freq_ge": f"readout_groups.{group}.members.{target}.frequency_mhz"},
    )


__all__ = ["analyze", "plot", "updates"]
