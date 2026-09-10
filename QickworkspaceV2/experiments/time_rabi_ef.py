"""TimeRabiEF: parameters, native program, analysis and registration in one file."""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces


class TimeRabiEFParameters(Parameters):
    start: float = Field(default=0.02, gt=0)
    stop: float = Field(default=2.0, gt=0)
    points: int = Field(default=81, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class TimeRabiEFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=False)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_length_ef"])
        # EF experiments independently own their GE state preparation.
        for q in cfg["targets"]:
            self.setup_qubit_gen(cfg["qubits"][q], "ge")
            self.setup_standard_gates(cfg["qubits"][q], "ge")
        for q in cfg["targets"]:
            self.setup_qb_pulse(cfg["qubits"][q], "ef", name="qb_pulse", pulse_type="const")

    def _body(self, cfg):
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        for q in cfg["targets"]:
            qc = cfg["qubits"][q]
            self.pulse(ch=qc["qb_ch_ef"], name=self.pulse_name(qc, "qb_pulse"), t=0)
        self.delay_auto(0.05)
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config("ef")
    cfg["steps"] = p.points
    axis = Sweep("length", p.start, p.stop, p.points, "us", "sweep", "length", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["pulse_type_ef"] = "const"
        qc["qb_length_ef"] = axis.qick()
    return ProgramPlan(TimeRabiEFProgram, cfg, (axis,))


def analyze(result):
    fits = analyze_traces(result, "rabi", signal=result.metadata["iq_process"])
    for fit in fits.values():
        for old, new in (("pi_gain", "pi_length_us"), ("pi2_gain", "pi2_length_us")):
            if old in fit.parameters:
                fit.parameters[new] = fit.parameters.pop(old)
                fit.errors[new] = fit.errors.pop(old)
                fit.units.pop(old, None)
                fit.units[new] = "us"
    return fits


experiment = ExperimentSpec(
    "time_rabi_ef",
    TimeRabiEFParameters,
    build,
    analyze,
    description="TimeRabiEF: independently maintained native EF protocol",
)
