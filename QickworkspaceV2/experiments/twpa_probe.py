"""Broadband probe for external-pump/flux TWPA procedures; absolute QICK MHz."""

from pydantic import Field, model_validator
from QickworkspaceV2 import BaseProgram, Parameters, ProgramPlan, ExperimentSpec


class TWPAProbeParameters(Parameters):
    start_mhz: float = 4000.0
    stop_mhz: float = 8000.0
    points: int = Field(default=101, ge=2, le=10001, strict=True)
    gain_scale: float = Field(default=0.1, gt=0, le=5)

    @model_validator(mode="after")
    def ordered(self):
        if self.start_mhz >= self.stop_mhz:
            raise ValueError("start_mhz must be below stop_mhz")
        return self


class TWPAProbeProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg)

    def _body(self, cfg):
        self.measure(cfg)


def build(ctx, p):
    if len(ctx.targets) != 1:
        raise ValueError("A TWPA probe uses one explicitly selected signal/readout path")
    cfg = ctx.config()
    qc = cfg["qubits"][ctx.targets[0]]
    center = qc["res_freq_ge"]
    qc["res_gain_ge"] *= p.gain_scale
    cfg["readout_groups"][qc["readout_group"]]["members"][ctx.targets[0]]["gain"] *= p.gain_scale
    return ProgramPlan(
        TWPAProbeProgram,
        cfg,
        metadata={
            "host_sweep": {
                "kind": "readout_frequency",
                "name": "frequency",
                "unit": "MHz",
                "start": p.start_mhz - center,
                "stop": p.stop_mhz - center,
                "points": p.points,
            }
        },
    )


experiment = ExperimentSpec(
    "twpa_probe",
    TWPAProbeParameters,
    build,
    description="Broadband complex transmission; compare matched pump-on/off reference runs",
)
