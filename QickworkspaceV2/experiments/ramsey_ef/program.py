"""ramsey_ef: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class RamseyEFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_sweep_loop(cfg, cfg["wait_us"])
        # EF experiments independently own their GE state preparation.
        for q in cfg["targets"]:
            self.setup_qubit_gen(cfg["qubits"][q], "ge")
            self.setup_standard_gates(cfg["qubits"][q], "ge")
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            phase = qc["qb_phase_ef"] + 360 * cfg["detuning_mhz"] * cfg["wait_us"]
            self.setup_qb_pulse(qc, "ef", name="analysis90", phase=phase, gain_key="pi2_gain_ef")

    def _body(self, cfg):
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).halfx()
        self.delay_auto(cfg["wait_us"], tag="evolution")
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch_ef"], name=self.pulse_name(qc, "analysis90"), t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg["steps"] = p.points
    axis = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg["wait_us"] = axis.qick()
    cfg["detuning_mhz"] = p.detuning_mhz
    return ProgramPlan(RamseyEFProgram, cfg, (axis,))
