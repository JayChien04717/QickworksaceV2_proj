from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.analysis import analyze_traces


class ResonatorSpecFParameters(Parameters):
    start: float = -3.0
    stop: float = 3.0
    points: int = Field(default=81, ge=8, le=10001, strict=True)
    gain_scale: float = Field(default=1.0, ge=0, le=5)

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class ResonatorSpecFProgram(BaseProgram):
    TRANSITION = "ef"

    def _initialize(self, cfg):
        # Mux tone registers are static: the recipe uses the common host-sweep runner.
        self.setup_device(cfg, gates=True)
        frequency = cfg["qubits"][cfg["targets"][0]]["res_freq_ge"]
        if getattr(frequency, "spans", {}):
            self.add_sweep_loop(cfg, frequency, "freqloop")
        for qc in cfg["qubits"].values():
            self.setup_qubit_gen(qc, "ge")
            self.setup_standard_gates(qc, "ge")

    def _body(self, cfg):
        for qc in cfg["qubits"].values():
            self.pulse(ch=qc["qb_ch"], name=self.pulse_name(qc, "x_ge"), t=0)
        self.wait_pulses()
        with self.parallel():
            for qb in self.qubits.values():
                qb.x()
        self.measure(cfg)


def build_resonator(ctx, p):
    cfg = ctx.config("ef")
    for qc in cfg["qubits"].values():
        qc["res_gain_ge"] *= p.gain_scale
    for group in cfg["readout_groups"].values():
        for q, member in group["members"].items():
            if q in cfg["targets"]:
                member["gain"] *= p.gain_scale
    # Host sweep works on mux/PYNQ readouts as well as tProc-controlled readouts.
    return ProgramPlan(
        ResonatorSpecFProgram,
        cfg,
        metadata={
            "host_sweep": {
                "kind": "readout_frequency",
                "name": "frequency",
                "unit": "MHz",
                "start": p.start,
                "stop": p.stop,
                "points": p.points,
            }
        },
    )


def analyze(result):
    return analyze_traces(result, "lorentzian", signal=result.metadata["iq_process"])


experiment = ExperimentSpec(
    "resonator_spec_f",
    ResonatorSpecFParameters,
    build_resonator,
    analyze,
    description="Readout resonance after preparing |f>; direct and mux",
)
