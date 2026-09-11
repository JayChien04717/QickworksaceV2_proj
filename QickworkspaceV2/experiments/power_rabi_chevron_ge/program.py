"""power_rabi_chevron_ge: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class PowerRabiChevronGEProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_gain_ge"])
        for q in cfg["targets"]:
            self.setup_qb_pulse(cfg["qubits"][q], "ge", name="qb_pulse")

    def _body(self, cfg):
        for _ in range(cfg["iterations"]):
            for q in cfg["targets"]:
                qc = cfg["qubits"][q]
                self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "qb_pulse"), t=0)
            self.delay_auto(0.02)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg["steps"] = p.points
    cfg["iterations"] = p.iterations
    axis = Sweep("gain", p.start, p.stop, p.points, "normalized gain", "sweep", "gain", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["qb_gain_ge"] = axis.qick()
    return ProgramPlan(PowerRabiChevronGEProgram, cfg, (axis,))
