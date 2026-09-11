"""state_tomography: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from itertools import product


class StateTomographyProgram(BaseProgram):
    TRANSITION = "ge"
    CAPTURE_SHOTS = True

    def _initialize(self, cfg):
        bases = list(product("XYZ", repeat=len(cfg["targets"])))
        prep = {"ground": [], "excited": ["x"], "plus": ["halfy"]}[cfg.get("preparation", "plus")]
        rotation = {"X": ["mhalfy"], "Y": ["halfx"], "Z": []}
        cfg.setdefault(
            "sequences", [{q: prep + rotation[b] for q, b in zip(cfg["targets"], basis)} for basis in bases]
        )
        self.READOUT_EVENTS = tuple("".join(basis) for basis in bases)
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)

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
