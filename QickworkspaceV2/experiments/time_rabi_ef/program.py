"""time_rabi_ef: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class TimeRabiEFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_length_ef"])
        # EF experiments independently own their GE state preparation.
        for q in cfg["targets"]:
            self.setup_qubit_gen(cfg["qubits"][q], "ge")
            self.setup_standard_gates(cfg["qubits"][q], "ge")
        for q in cfg["targets"]:
            self.setup_qb_pulse(cfg["qubits"][q], "ef", name="qb_pulse", pulse_type="const")

    def _body(self, cfg):
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch_ef"], name=self.pulse_name(qc, "qb_pulse"), t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg["steps"] = p.points
    axis = Sweep("length", p.start, p.stop, p.points, "us", "sweep", "length", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["pulse_type_ef"] = "const"
        qc["qb_length_ef"] = axis.qick()
    return ProgramPlan(TimeRabiEFProgram, cfg, (axis,))
