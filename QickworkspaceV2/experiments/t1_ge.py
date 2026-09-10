"""T1GE: parameters, native program, analysis and registration in one file."""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces


class T1GEParameters(Parameters):
    start: float = Field(default=0.0, ge=0)
    stop: float = Field(default=100.0, gt=0)
    points: int = Field(default=81, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class T1GEProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)
        self.add_sweep_loop(cfg, cfg["wait_us"])

    def _body(self, cfg):
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).x()
        self.delay_auto(cfg["wait_us"], tag="evolution")
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg["steps"] = p.points
    axis = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg["wait_us"] = axis.qick()
    return ProgramPlan(T1GEProgram, cfg, (axis,))


def analyze(result):
    fits = analyze_traces(result, "exponential", signal=result.metadata["iq_process"])
    return fits


experiment = ExperimentSpec(
    "t1_ge", T1GEParameters, build, analyze, description="T1GE: independently maintained native GE protocol"
)
