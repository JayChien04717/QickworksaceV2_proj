"""spin_echo_ge: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class SpinEchoGEProgram(BaseProgram):
    AXIS_SCALE = 2
    TRANSITION = "ge"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_sweep_loop(cfg, cfg["wait_us"])

    def _body(self, cfg):
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).halfx()
        self.delay_auto(cfg["wait_us"] / 2, tag="evolution")
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).x()
        self.delay_auto(cfg["wait_us"] / 2)
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).halfx()
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg["steps"] = p.points
    axis = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg["wait_us"] = axis.qick()
    return ProgramPlan(SpinEchoGEProgram, cfg, (axis,), metadata={"axis_scale": 2})
