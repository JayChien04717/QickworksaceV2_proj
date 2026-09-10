"""RamseyGE: parameters, native program, analysis and registration in one file."""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces


class RamseyGEParameters(Parameters):
    start: float = Field(default=0.0, ge=0)
    stop: float = Field(default=30.0, gt=0)
    detuning_mhz: float = 0.2
    points: int = Field(default=81, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class RamseyGEProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)
        self.add_sweep_loop(cfg, cfg["wait_us"])
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            phase = qc["qb_phase_ge"] + 360 * cfg["detuning_mhz"] * cfg["wait_us"]
            self.setup_qb_pulse(qc, "ge", name="analysis90", phase=phase, gain_key="pi2_gain_ge")

    def _body(self, cfg):
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).halfx()
        self.delay_auto(cfg["wait_us"], tag="evolution")
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "analysis90"), t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg["steps"] = p.points
    axis = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg["wait_us"] = axis.qick()
    cfg["detuning_mhz"] = p.detuning_mhz
    return ProgramPlan(RamseyGEProgram, cfg, (axis,))


def analyze(result):
    fits = analyze_traces(result, "ramsey", signal=result.metadata["iq_process"])
    return fits


experiment = ExperimentSpec(
    "ramsey_ge",
    RamseyGEParameters,
    build,
    analyze,
    description="RamseyGE: independently maintained native GE protocol",
)
