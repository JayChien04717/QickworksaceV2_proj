"""PowerRabiEF: parameters, native program, analysis and registration in one file."""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.analysis import analyze_traces


class PowerRabiEFParameters(Parameters):
    start: float = Field(default=0.0, ge=0, le=1)
    stop: float = Field(default=0.6, gt=0, le=1)
    points: int = Field(default=81, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class PowerRabiEFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=False)
        self.add_sweep_loop(cfg, cfg["qubits"][cfg["targets"][0]]["qb_gain_ef"])
        # EF experiments independently own their GE state preparation.
        for q in cfg["targets"]:
            self.setup_qubit_gen(cfg["qubits"][q], "ge")
            self.setup_standard_gates(cfg["qubits"][q], "ge")
        for q in cfg["targets"]:
            self.setup_qb_pulse(cfg["qubits"][q], "ef", name="qb_pulse")

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
    axis = Sweep("gain", p.start, p.stop, p.points, "normalized gain", "sweep", "gain", "{target}__qb_pulse")
    for qc in cfg["qubits"].values():
        qc["qb_gain_ef"] = axis.qick()
    return ProgramPlan(PowerRabiEFProgram, cfg, (axis,))


def analyze(result):
    fits = analyze_traces(result, "rabi", signal=result.metadata["iq_process"])
    return fits


experiment = ExperimentSpec(
    "power_rabi_ef",
    PowerRabiEFParameters,
    build,
    analyze,
    description="PowerRabiEF: independently maintained native EF protocol",
)
