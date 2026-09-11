"""conditional_ramsey: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan
from QickworkspaceV2.programs.sweeps import Sweep


class ConditionalRamseyProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)
        self.add_loop("sweep", cfg["steps"])
        qc = cfg["qubits"][cfg["targets"][1]]
        self.setup_qb_pulse(
            qc, name="analysis90", phase=360 * cfg["detuning_mhz"] * cfg["wait_us"], gain_key="pi2_gain_ge"
        )

    def _body(self, cfg):
        control, target = cfg["targets"]
        qc = cfg["qubits"][target]
        self.delay_auto(cfg["reset_wait_us"])
        for state in (0, 1):
            if state:
                self.delay_auto(cfg["reset_wait_us"])
                self.qubit(control).x()
            self.qubit(target).halfx()
            self.delay_auto(cfg["wait_us"], tag="evolution" if state == 0 else "evolution_e")
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "analysis90"), t=0)
            self.delay_auto(0.05)
            self.measure(cfg)


def build_conditional(ctx, p):
    if len(ctx.targets) != 2:
        raise ValueError("conditional_ramsey needs exactly two GE targets: control,target")
    cfg = ctx.config()
    cfg["steps"] = p.points
    sweep = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg.update(wait_us=sweep.qick(), detuning_mhz=p.detuning_mhz, reset_wait_us=p.reset_wait_us)
    return ProgramPlan(
        ConditionalRamseyProgram,
        cfg,
        (sweep,),
        ("control_g", "control_e"),
    )
