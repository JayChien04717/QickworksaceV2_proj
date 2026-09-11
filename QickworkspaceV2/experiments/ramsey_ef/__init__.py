"""ramsey_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import RamseyEFParameters
from .program import RamseyEFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["RamseyEFParameters", "RamseyEFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "ramsey_ef",
    RamseyEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="RamseyEF: independently maintained native EF protocol",
)

RamseyEFProgram.EXPERIMENT = experiment
