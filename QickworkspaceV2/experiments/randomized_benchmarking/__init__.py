"""randomized_benchmarking: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResetParameters, RBParameters
from .program import RBProgram, build_rb
from .analysis import analyze_rb, plot, updates

__all__ = [
    "ResetParameters",
    "RBParameters",
    "RBProgram",
    "build_rb",
    "analyze_rb",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "randomized_benchmarking",
    RBParameters,
    build_rb,
    analyze_rb,
    plot=plot,
    updates=updates,
    description="randomized benchmarking: independent native protocol",
)

RBProgram.EXPERIMENT = experiment
