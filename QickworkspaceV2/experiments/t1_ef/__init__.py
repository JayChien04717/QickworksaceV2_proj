"""t1_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import T1EFParameters
from .program import T1EFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["T1EFParameters", "T1EFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "t1_ef",
    T1EFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="T1EF: independently maintained native EF protocol",
)

T1EFProgram.EXPERIMENT = experiment
