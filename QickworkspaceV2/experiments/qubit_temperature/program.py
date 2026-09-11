"""qubit_temperature: program."""

from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.experiments.base import ProgramPlan


class QubitTemperatureProgram(BaseProgram):
    TRANSITION = "ef"
    READOUT_EVENTS = ("thermal", "ge_prepared")

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_gain_ef"])
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, "ge")
            self.setup_standard_gates(qc, "ge")
            self.setup_qb_pulse(qc, "ef", name="qb_pulse")

    def _body(self, cfg):
        for prepared in (False, True):
            self.delay_auto(cfg["reset_wait_us"])
            if prepared:
                for qc in cfg["qubits"].values():
                    self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
                self.wait_pulses()
            for qc in cfg["qubits"].values():
                self.pulse(ch=qc["qb_ch_ef"], name=self.pulse_name(qc, "qb_pulse"), t=0)
            self.wait_pulses()
            self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg.update(steps=p.points, reset_wait_us=p.reset_wait_us)
    axis = Sweep("gain", p.start, p.stop, p.points, "normalized gain", "sweep", "gain", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["qb_gain_ef"] = axis.qick()
    return ProgramPlan(QubitTemperatureProgram, cfg, (axis,), QubitTemperatureProgram.READOUT_EVENTS)
