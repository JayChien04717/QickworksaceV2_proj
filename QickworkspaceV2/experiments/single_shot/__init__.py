"""single_shot: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResetParameters
from .program import SingleShotProgram, build_single_shot
from .analysis import analyze, plot, updates

__all__ = [
    "ResetParameters",
    "SingleShotProgram",
    "build_single_shot",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "single_shot",
    ResetParameters,
    build_single_shot,
    analyze,
    plot=plot,
    updates=updates,
    description="Paired ground/excited shots with held-out discrimination",
)

SingleShotProgram.EXPERIMENT = experiment
