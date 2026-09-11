"""state_tomography: analysis."""

from __future__ import annotations

from QickworkspaceV2.calibration.updates import CalibrationUpdates
import numpy as np
from itertools import product
from QickworkspaceV2.data.models import FitResult
from matplotlib.figure import Figure
from QickworkspaceV2.plotting.plots import plot_result


def analyze_tomography(result):
    targets = result.metadata["targets"]
    bases = [list(str(label)) for label in result[targets[0]].coords["readout"]]
    expected_bases = list(product("XYZ", repeat=len(targets)))
    if [tuple(b) for b in bases] != expected_bases:
        raise ValueError("Tomography requires every ordered XYZ product basis in the readout labels")
    states = []
    for q in targets:
        trace = result[q]
        if trace.shots is None or trace.shot_dims != ("shot", "readout"):
            raise ValueError("Tomography requires aligned joint shots")
        threshold = trace.metadata.get("threshold")
        if threshold is None or not np.isfinite(threshold):
            raise ValueError(f"{q}: finite readout threshold required")
        states.append(
            (trace.shots * np.exp(-1j * np.deg2rad(trace.metadata["rotation_deg"]))).real > threshold
        )
    shots = np.stack(states, axis=-1)
    paulis = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    rho = np.zeros((2 ** len(targets),) * 2, complex)
    expectations = {}
    for term in product("IXYZ", repeat=len(targets)):
        compatible = [
            i for i, basis in enumerate(bases) if all(p == "I" or p == b for p, b in zip(term, basis))
        ]
        selected = [i for i, p in enumerate(term) if p != "I"]
        estimate = (
            float(np.mean(np.prod(1 - 2 * shots[:, compatible, :][:, :, selected].astype(float), axis=-1)))
            if selected
            else 1.0
        )
        matrix = np.array([[1]], complex)
        for p in term:
            matrix = np.kron(matrix, paulis[p])
        rho += estimate * matrix / (2 ** len(targets))
        expectations["".join(term)] = estimate
    eigenvalues = np.linalg.eigvalsh(rho)
    result.metadata["tomography"] = {
        "density_matrix": rho,
        "pauli_expectations": expectations,
        "method": "linear inversion without assignment correction or positivity projection",
    }
    return {
        "joint": FitResult(
            "state_tomography",
            bool(eigenvalues.min() > -0.15),
            {"purity": float(np.trace(rho @ rho).real), "minimum_eigenvalue": float(eigenvalues.min())},
            message="Linear inversion; finite-shot estimates can have negative eigenvalues",
        )
    }


def plot(result, **options):
    record = result.metadata.get("tomography")
    if record is None:
        return plot_result(result, **options)
    rho = np.asarray(record["density_matrix"], dtype=object)
    rho = np.vectorize(
        lambda value: complex(value["real"], value["imag"]) if isinstance(value, dict) else complex(value)
    )(rho)
    figure = Figure(figsize=(9, 4), layout="constrained")
    for ax, values, title in zip(figure.subplots(1, 2), [rho.real, rho.imag], ["Re rho", "Im rho"]):
        image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        figure.colorbar(image, ax=ax)
    return figure


def updates(result, target):
    """Declare this experiment's explicit calibration updates; never write files."""
    return CalibrationUpdates(reason="This diagnostic reports metrics without automatic calibration updates")


__all__ = ["analyze_tomography", "plot", "updates"]
