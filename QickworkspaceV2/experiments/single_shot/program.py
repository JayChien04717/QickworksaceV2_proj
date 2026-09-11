"""single_shot: program."""

from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import ProgramPlan


class SingleShotProgram(BaseProgram):
    TRANSITION = "ge"
    READOUT_EVENTS = ("ground", "excited")
    CAPTURE_SHOTS = True

    def _initialize(self, cfg):
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, cfg["transition"])
            self.setup_standard_gates(qc, cfg["transition"])
        self.setup_readout(cfg)

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        self.measure(cfg)
        self.delay_auto(cfg["reset_wait_us"])
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)


def build_single_shot(ctx, p):
    cfg = ctx.config()
    cfg["reset_wait_us"] = p.reset_wait_us
    cfg["classifier_axis"] = p.classifier_axis
    if cfg["reps"] < 40:
        raise ValueError("Single-shot discrimination needs at least 40 reps")
    return ProgramPlan(
        SingleShotProgram,
        cfg,
        readout_events=("ground", "excited"),
        capture_shots=True,
    )
