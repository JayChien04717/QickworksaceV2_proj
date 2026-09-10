from __future__ import annotations
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec


class TimeOfFlightParameters(Parameters):
    """Targets for the decimated readout diagnostic."""


class TimeOfFlightProgram(BaseProgram):
    ACQUISITION_MODE = "decimated"
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg)

    def _body(self, cfg):
        self.measure(cfg)


def build_tof(ctx, p):
    cfg = ctx.config()
    cfg["reps"] = 1
    return ProgramPlan(
        TimeOfFlightProgram,
        cfg,
        metadata={
            "acquisition_mode": "decimated",
            "note": "One hardware rep per software average to fit the decimated buffer",
        },
    )


experiment = ExperimentSpec(
    "time_of_flight",
    TimeOfFlightParameters,
    build_tof,
    description="Raw decimated I/Q; one hardware rep and software averaging",
)
