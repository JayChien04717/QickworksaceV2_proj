from __future__ import annotations
import numpy as np
from pydantic import Field
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec

from QickworkspaceV2.data.models import FitResult

ALLXY = [
    ("i", "i"),
    ("x", "x"),
    ("y", "y"),
    ("x", "y"),
    ("y", "x"),
    ("halfx", "i"),
    ("halfy", "i"),
    ("halfx", "halfy"),
    ("halfy", "halfx"),
    ("halfx", "y"),
    ("halfy", "x"),
    ("x", "halfy"),
    ("y", "halfx"),
    ("halfx", "x"),
    ("x", "halfx"),
    ("halfy", "y"),
    ("y", "halfy"),
    ("x", "i"),
    ("y", "i"),
    ("halfx", "halfx"),
    ("halfy", "halfy"),
]

ALLXY_IDEAL = [0.0] * 5 + [0.5] * 12 + [1.0] * 4


class ResetParameters(Parameters):
    reset_wait_us: float = Field(default=200.0, gt=0)


class AllXYProgram(BaseProgram):
    TRANSITION = "ge"

    def _initialize(self, cfg):
        self.setup_device(cfg, gates=True)

    def _body(self, cfg):
        self.delay_auto(cfg["reset_wait_us"])
        for index, sequences in enumerate(cfg["sequences"]):
            if index:
                self.delay_auto(cfg["reset_wait_us"])
            for layer in range(max(map(len, sequences.values()))):
                with self.parallel():
                    for q, sequence in sequences.items():
                        if layer < len(sequence) and sequence[layer] != "i":
                            getattr(self.qubit(q), sequence[layer])()
            self.measure(cfg)


def build_allxy(ctx, p):
    cfg = ctx.config()
    cfg["reset_wait_us"] = p.reset_wait_us
    cfg["sequences"] = [{q: gates for q in ctx.targets} for gates in ALLXY]
    cfg["ideal_population"] = ALLXY_IDEAL
    return ProgramPlan(
        AllXYProgram,
        cfg,
        readout_events=tuple(f"{a},{b}" for a, b in ALLXY),
        metadata={"ideal_population": ALLXY_IDEAL},
    )


def analyze_allxy(result):
    fits = {}
    for q, trace in result.traces.items():
        y = trace.signal(result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0))
        ground, excited = float(np.mean(y[:5])), float(np.mean(y[-4:]))
        if abs(excited - ground) < 1e-10:
            fits[q] = FitResult("allxy", False, message="No state contrast")
            continue
        population = (y - ground) / (excited - ground)
        error = float(np.sqrt(np.mean((population - np.asarray(ALLXY_IDEAL)) ** 2)))
        fits[q] = FitResult(
            "allxy",
            error < 0.1,
            {"normalized_rms_error": error},
            message="Endpoint-normalized AllXY; not a gate fidelity estimate",
        )
    return fits


experiment = ExperimentSpec(
    "allxy", ResetParameters, build_allxy, analyze_allxy, description="allxy: independent native protocol"
)
