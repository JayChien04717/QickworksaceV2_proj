from __future__ import annotations
from pydantic import Field
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec

from QickworkspaceV2.analysis import analyze_single_shot


class ResetParameters(Parameters):
    reset_wait_us: float = Field(default=200.0, gt=0)


class SingleShotProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)

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
    if cfg["reps"] < 40:
        raise ValueError("Single-shot discrimination needs at least 40 reps")
    return ProgramPlan(
        SingleShotProgram,
        cfg,
        readout_events=("ground", "excited"),
        capture_shots=True,
    )


def analyze(result):
    return analyze_single_shot(result)


experiment = ExperimentSpec(
    "single_shot",
    ResetParameters,
    build_single_shot,
    analyze,
    description="Paired ground/excited shots with held-out discrimination",
)
