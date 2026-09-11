"""power_rabi_chevron_ef: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class PowerRabiChevronEFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_gain_ef"])
        # EF experiments independently own their GE state preparation.
        for q in cfg["targets"]:
            self.setup_qubit_gen(cfg["qubits"][q], "ge")
            self.setup_standard_gates(cfg["qubits"][q], "ge")
        for q in cfg["targets"]:
            self.setup_qb_pulse(cfg["qubits"][q], "ef", name="qb_pulse")

    def _body(self, cfg):
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        for _ in range(cfg["iterations"]):
            for q in cfg["targets"]:
                qc = cfg["qubits"][q]
                self.pulse(ch=qc["qb_ch_ef"], name=self.pulse_name(qc, "qb_pulse"), t=0)
            self.delay_auto(0.02)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg["steps"] = p.points
    cfg["iterations"] = p.iterations
    axis = Sweep("gain", p.start, p.stop, p.points, "normalized gain", "sweep", "gain", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["qb_gain_ef"] = axis.qick()
    return ProgramPlan(PowerRabiChevronEFProgram, cfg, (axis,))
