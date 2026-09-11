"""coupler_chevron: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class ChevronProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_loop("gainloop", cfg["gain_points"])
        self.add_loop("lengthloop", cfg["length_points"])
        self.setup_coupler(cfg, cfg["edge"])

    def _body(self, cfg):
        self.qubit(cfg["targets"][0]).x()
        self.interaction(cfg["edge"])
        self.measure(cfg)


def build_chevron(ctx, p):
    cfg = ctx.config()
    if (
        p.coupler not in cfg["couplers"]
        or set(cfg["couplers"][p.coupler]["qubits"]) != set(ctx.targets)
        or len(ctx.targets) != 2
    ):
        raise ValueError("Select the two endpoints of the named coupler")
    edge = cfg["couplers"][p.coupler]
    if edge["pulse"]["style"] == "arb":
        raise ValueError("Chevron duration sweep needs const or flat_top coupler pulse")
    gain = Sweep(
        "gain",
        p.gain_start,
        p.gain_stop,
        p.gain_points,
        "normalized gain",
        "gainloop",
        "gain",
        p.coupler + "__interaction",
    )
    length = Sweep(
        "length",
        p.length_start_us,
        p.length_stop_us,
        p.length_points,
        "us",
        "lengthloop",
        "length",
        p.coupler + "__interaction",
    )
    edge["pulse"]["gain"], edge["pulse"]["length_us"] = gain.qick(), length.qick()
    cfg.update(edge=p.coupler, gain_points=p.gain_points, length_points=p.length_points)
    return ProgramPlan(
        ChevronProgram,
        cfg,
        (gain, length),
        metadata={"interpretation": "Interaction population map; does not certify an entangling gate"},
    )
