"""Readout spectrum with an explicit QICK bias pulse; scan bias_gain outside."""

from pydantic import Field, model_validator
from QickworkspaceV2 import BaseProgram, Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.analysis import analyze_traces


class ResonatorFluxParameters(Parameters):
    bias_port: str = Field(description="Logical generator ID connected to the flux/bias line")
    bias_gain: float = Field(default=0.0, ge=-1, le=1)
    settle_us: float = Field(default=0.1, ge=0)
    bias_length_us: float = Field(default=10.0, gt=0)
    start: float = -3.0
    stop: float = 3.0
    points: int = Field(default=41, ge=8, le=10001, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self


class ResonatorFluxProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg)
        port = cfg["bias_port"]
        self.declare_gen_auto(port["channel"], port["nqz"], "mixer_mhz", port)
        self.add_pulse(
            ch=port["channel"],
            name="bias",
            style="const",
            freq=0,
            gain=cfg["bias_gain"],
            phase=0,
            length=cfg["bias_length_us"],
        )

    def _body(self, cfg):
        self.pulse(ch=cfg["bias_port"]["channel"], name="bias", t=0)
        # Shift the time reference while the bias is still playing.
        self.delay(cfg["settle_us"])
        self.measure(cfg)


def build(ctx, p):
    cfg = ctx.config()
    port = ctx.device.hardware.generators[p.bias_port]
    cfg.update(
        bias_port=port.model_dump(),
        bias_gain=p.bias_gain,
        settle_us=p.settle_us,
        bias_length_us=p.bias_length_us,
    )
    cfg["auxiliary_ports"] = {p.bias_port: p.bias_gain}
    required = max(
        max(g["length_us"], g["trigger_us"] + g["integration_us"]) for g in cfg["readout_groups"].values()
    )
    if p.bias_length_us < p.settle_us + required:
        raise ValueError("Bias pulse must cover settling, readout and the full integration window")
    return ProgramPlan(
        ResonatorFluxProgram,
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
    "resonator_flux",
    ResonatorFluxParameters,
    build,
    analyze,
    description="Readout spectroscopy under a timed QICK bias pulse",
)
