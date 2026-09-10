from __future__ import annotations
import numpy as np
from pydantic import Field
from QickworkspaceV2.programs.base import BaseProgram
from QickworkspaceV2.experiments.base import Parameters, ProgramPlan, ExperimentSpec

from scipy.optimize import curve_fit
from QickworkspaceV2.data.models import FitResult
from QickworkspaceV2.programs.cliffords import randomized_sequence


class ResetParameters(Parameters):
    reset_wait_us: float = Field(default=200.0, gt=0)


class RBParameters(ResetParameters):
    depths: str = Field(
        default="0,1,2,4,8,12,16,24", description="Comma-separated nonnegative Clifford counts"
    )
    sequences_per_depth: int = Field(default=2, ge=2, le=20, strict=True)
    seed: int = Field(default=42, ge=0, strict=True)


class RBProgram(BaseProgram):
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


def build_rb(ctx, p):
    depths = [int(x.strip()) for x in p.depths.split(",")]
    if len(depths) < 5 or len(set(depths)) != len(depths) or min(depths) < 0 or max(depths) > 1000:
        raise ValueError("RB needs at least five distinct depths between 0 and 1000")
    cfg = ctx.config()
    rng = np.random.default_rng(p.seed)
    cfg["reset_wait_us"] = p.reset_wait_us
    cfg["sequences"] = [
        {q: randomized_sequence(depth, rng) for q in ctx.targets}
        for depth in depths
        for repeat in range(p.sequences_per_depth)
    ]
    cfg["rb_depths"] = list(np.repeat(depths, p.sequences_per_depth).astype(int))
    if sum(max(map(len, seq.values())) for seq in cfg["sequences"]) > 2000:
        raise ValueError("RB program is too large; reduce depths/sequences or partition the acquisition")
    events = tuple(f"m{depth}_s{repeat}" for depth in depths for repeat in range(p.sequences_per_depth))
    return ProgramPlan(
        RBProgram,
        cfg,
        readout_events=events,
        metadata={"rb_depths": depths, "sequences_per_depth": p.sequences_per_depth, "seed": p.seed},
    )


def analyze_rb(result):
    depths = np.asarray(result.metadata["rb_depths"])
    repeats = result.metadata["sequences_per_depth"]

    def model(x, offset, amplitude, probability):
        return offset + amplitude * probability**x

    fits = {}
    for q, trace in result.traces.items():
        values = trace.signal(
            result.metadata["iq_process"], rotation_deg=trace.metadata.get("rotation_deg", 0)
        ).reshape(len(depths), repeats)
        y = values.mean(axis=1)
        try:
            params, cov = curve_fit(
                model,
                depths,
                y,
                p0=[float(y[-1]), float(y[0] - y[-1]), 0.99],
                bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, 1]),
                maxfev=20000,
            )
            err = np.sqrt(np.maximum(np.diag(cov), 0))
            residual = y - model(depths, *params)
            total = np.sum((y - y.mean()) ** 2)
            r2 = float(1 - np.sum(residual**2) / total) if total > 0 else 0
            good = np.isfinite(cov).all() and r2 > 0.8 and err[2] < max(1 - params[2], 1e-8)
            fits[q] = FitResult(
                "rb",
                bool(good),
                {"decay_probability": float(params[2]), "error_per_clifford": float((1 - params[2]) / 2)},
                {"decay_probability": float(err[2]), "error_per_clifford": float(err[2] / 2)},
                r_squared=r2,
                message="Unweighted fit of sequence means; EPC assumes Markovian single-qubit Clifford noise",
            )
        except (ValueError, RuntimeError) as exc:
            fits[q] = FitResult("rb", False, message=str(exc))
    return fits


experiment = ExperimentSpec(
    "randomized_benchmarking",
    RBParameters,
    build_rb,
    analyze_rb,
    description="randomized benchmarking: independent native protocol",
)
