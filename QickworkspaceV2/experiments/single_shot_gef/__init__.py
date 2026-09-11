"""single_shot_gef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import SingleShotGEFParameters
from .program import SingleShotGEFProgram, build
from .analysis import analyze, plot, updates

__all__ = [
    "SingleShotGEFParameters",
    "SingleShotGEFProgram",
    "build",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "single_shot_gef",
    SingleShotGEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Prepared g/e/f raw shots and held-out three-state classification",
)

SingleShotGEFProgram.EXPERIMENT = experiment
