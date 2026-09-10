from __future__ import annotations
import numpy as np
from pydantic import Field
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec

from itertools import product
from QickworkspaceV2.data.models import FitResult


class ResetParameters(Parameters):
    reset_wait_us: float = Field(default=200.0, gt=0)


class TomographyParameters(ResetParameters):
    preparation: str = Field(
        default="plus",
        description="ground, excited or plus; custom state preparation belongs in a custom program",
    )


class StateTomographyProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        for index, sequences in enumerate(cfg["sequences"]):
            if index:
                self.delay_auto(cfg["reset_wait_us"])
            for layer in range(max(map(len, sequences.values()))):
                with self.parallel():
                    for q, sequence in sequences.items():
                        if layer < len(sequence) and sequence[layer] != "i":
                            getattr(self.qubit(q), sequence[layer])()
            self.measure(cfg)


def build_tomography(ctx, p):
    if len(ctx.targets) > 3:
        raise ValueError(
            "Full tomography is limited to three qubits (3^N bases); use selected observables for larger devices"
        )
    if p.preparation not in ("ground", "excited", "plus"):
        raise ValueError("Unknown state preparation")
    cfg = ctx.config()
    for q in ctx.targets:
        if ctx.device.readout(q)[2].threshold is None:
            raise ValueError(f"{q}: calibrate and commit single-shot threshold/rotation before tomography")
    bases = list(product("XYZ", repeat=len(ctx.targets)))
    prep = {"ground": [], "excited": ["x"], "plus": ["halfy"]}[p.preparation]
    rotation = {"X": ["mhalfy"], "Y": ["halfx"], "Z": []}
    cfg["reset_wait_us"] = p.reset_wait_us
    cfg["sequences"] = [{q: prep + rotation[b] for q, b in zip(ctx.targets, basis)} for basis in bases]
    cfg["tomography_preparation"] = p.preparation
    return ProgramPlan(
        StateTomographyProgram,
        cfg,
        readout_events=tuple("".join(b) for b in bases),
        capture_shots=True,
        metadata={"tomography_bases": [list(b) for b in bases]},
    )


def analyze_tomography(result):
    from QickworkspaceV2.device import Device

    device = Device(
        **{
            "hardware": result.metadata["device_snapshot"]["hardware"],
            "config": result.metadata["device_snapshot"]["device"],
        }
    )
    targets = result.metadata["targets"]
    bases = result.metadata["tomography_bases"]
    states = []
    for q in targets:
        trace = result[q]
        if trace.shots is None or trace.shot_dims != ("shot", "readout"):
            raise ValueError("Tomography requires aligned joint shots")
        member = device.readout(q)[2]
        states.append((trace.shots * np.exp(-1j * np.deg2rad(member.rotation_deg))).real > member.threshold)
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


experiment = ExperimentSpec(
    "state_tomography",
    TomographyParameters,
    build_tomography,
    analyze_tomography,
    description="state tomography: independent native protocol",
)
