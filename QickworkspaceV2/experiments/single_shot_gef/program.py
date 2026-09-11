"""single_shot_gef: program."""

from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan


class SingleShotGEFProgram(BaseProgram):
    TRANSITION = "ef"
    READOUT_EVENTS = ("ground", "excited", "second_excited")
    CAPTURE_SHOTS = True

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, "ge")
            self.setup_standard_gates(qc, "ge")

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        self.measure(cfg)
        for state in ("excited", "second_excited"):
            self.delay_auto(cfg["reset_wait_us"])
            for qc in cfg["qubits"].values():
                self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
            self.wait_pulses()
            if state == "second_excited":
                with self.parallel():
                    for qb in self.qubits.values():
                        qb.x()
            self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg["reset_wait_us"] = p.reset_wait_us
    if cfg["reps"] < 40:
        raise ValueError("Three-state classification requires at least 40 reps")
    return ProgramPlan(
        SingleShotGEFProgram, cfg, readout_events=("ground", "excited", "second_excited"), capture_shots=True
    )
