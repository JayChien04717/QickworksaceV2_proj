"""ramsey_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import RamseyGEParameters
from .program import RamseyGEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["RamseyGEParameters", "RamseyGEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "ramsey_ge",
    RamseyGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="RamseyGE: independently maintained native GE protocol",
)

RamseyGEProgram.EXPERIMENT = experiment
