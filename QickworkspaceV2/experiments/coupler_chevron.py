from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec
from QickworkspaceV2.programs.sweeps import Sweep


class ChevronParameters(Parameters):
    target: str = "Q1,Q2"
    coupler: str = "C12"
    gain_start: float = Field(default=0.01, ge=-1, le=1)
    gain_stop: float = Field(default=0.2, ge=-1, le=1)
    gain_points: int = Field(default=21, ge=2, le=501, strict=True)
    length_start_us: float = Field(default=0.02, gt=0)
    length_stop_us: float = Field(default=2.0, gt=0)
    length_points: int = Field(default=81, ge=2, le=1001, strict=True)

    @model_validator(mode="after")
    def ranges(self):
        if self.gain_start >= self.gain_stop or self.length_start_us >= self.length_stop_us:
            raise ValueError("Chevron starts must be below stops")
        return self


class ChevronProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)
        self.add_loop("gainloop", cfg["gain_points"])
        self.add_loop("lengthloop", cfg["length_points"])
        self.setup_coupler(cfg, cfg["edge"])

    def _body(self, cfg):
        self.qubit(cfg["targets"][0]).x()
        self.interaction(cfg["edge"])
        self.measure(cfg)


def build_chevron(ctx, p):
    cfg = ctx.config()
    if (
        p.coupler not in cfg["couplers"]
        or set(cfg["couplers"][p.coupler]["qubits"]) != set(ctx.targets)
        or len(ctx.targets) != 2
    ):
        raise ValueError("Select the two endpoints of the named coupler")
    edge = cfg["couplers"][p.coupler]
    if edge["pulse"]["style"] == "arb":
        raise ValueError("Chevron duration sweep needs const or flat_top coupler pulse")
    gain = Sweep(
        "gain",
        p.gain_start,
        p.gain_stop,
        p.gain_points,
        "normalized gain",
        "gainloop",
        "gain",
        p.coupler + "__interaction",
    )
    length = Sweep(
        "length",
        p.length_start_us,
        p.length_stop_us,
        p.length_points,
        "us",
        "lengthloop",
        "length",
        p.coupler + "__interaction",
    )
    edge["pulse"]["gain"], edge["pulse"]["length_us"] = gain.qick(), length.qick()
    cfg.update(edge=p.coupler, gain_points=p.gain_points, length_points=p.length_points)
    return ProgramPlan(
        ChevronProgram,
        cfg,
        (gain, length),
        metadata={"interpretation": "Interaction population map; does not certify an entangling gate"},
    )


experiment = ExperimentSpec(
    "coupler_chevron",
    ChevronParameters,
    build_chevron,
    description="Two-dimensional native coupler interaction scan",
)
