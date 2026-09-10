"""SpinEchoEF: parameters, native program, analysis and registration in one file."""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces


class SpinEchoEFParameters(Parameters):
    start: float = Field(default=0.0, ge=0)
    stop: float = Field(default=100.0, gt=0)
    points: int = Field(default=81, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class SpinEchoEFProgram(BaseProgram):
    AXIS_SCALE = 2
    TRANSITION = "ef"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)
        self.add_sweep_loop(cfg, cfg["wait_us"])
        # EF experiments independently own their GE state preparation.
        for q in cfg["targets"]:
            self.setup_qubit_gen(cfg["qubits"][q], "ge")
            self.setup_standard_gates(cfg["qubits"][q], "ge")

    def _body(self, cfg):
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).halfx()
        self.delay_auto(cfg["wait_us"] / 2, tag="evolution")
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).x()
        self.delay_auto(cfg["wait_us"] / 2)
        with self.parallel():
            for q in cfg["targets"]:
                self.qubit(q).halfx()
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg["steps"] = p.points
    axis = Sweep("delay", p.start, p.stop, p.points, "us", "sweep", "t", tag="evolution")
    cfg["wait_us"] = axis.qick()
    return ProgramPlan(SpinEchoEFProgram, cfg, (axis,), metadata={"axis_scale": 2})


def analyze(result):
    fits = analyze_traces(result, "exponential", signal=result.metadata["iq_process"])
    return fits


experiment = ExperimentSpec(
    "spin_echo_ef",
    SpinEchoEFParameters,
    build,
    analyze,
    description="SpinEchoEF: independently maintained native EF protocol",
)
