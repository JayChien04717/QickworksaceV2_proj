"""PowerRabiChevronGE: parameters, native program, analysis and registration in one file."""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces


class PowerRabiChevronGEParameters(Parameters):
    iterations: int = Field(default=3, ge=1, le=101, strict=True)
    start: float = Field(default=0.0, ge=0, le=1)
    stop: float = Field(default=0.6, gt=0, le=1)
    points: int = Field(default=81, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class PowerRabiChevronGEProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=False)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_gain_ge"])
        for q in cfg["targets"]:
            self.setup_qb_pulse(cfg["qubits"][q], "ge", name="qb_pulse")

    def _body(self, cfg):
        for _ in range(cfg["iterations"]):
            for q in cfg["targets"]:
                qc = cfg["qubits"][q]
                self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "qb_pulse"), t=0)
            self.delay_auto(0.02)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ge")
    cfg["steps"] = p.points
    cfg["iterations"] = p.iterations
    axis = Sweep("gain", p.start, p.stop, p.points, "normalized gain", "sweep", "gain", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["qb_gain_ge"] = axis.qick()
    return ProgramPlan(PowerRabiChevronGEProgram, cfg, (axis,))


def analyze(result):
    fits = analyze_traces(result, "rabi", signal=result.metadata["iq_process"])
    for fit in fits.values():
        for key in ("pi_gain", "pi2_gain"):
            if key in fit.parameters:
                fit.parameters[key] *= result.metadata["parameters"]["iterations"]
                if fit.errors.get(key) is not None:
                    fit.errors[key] *= result.metadata["parameters"]["iterations"]
        fit.message += "; repeated-pulse estimate, verify the single-pulse Rabi calibration"
    return fits


experiment = ExperimentSpec(
    "power_rabi_chevron_ge",
    PowerRabiChevronGEParameters,
    build,
    analyze,
    description="PowerRabiChevronGE: independently maintained native GE protocol",
)
