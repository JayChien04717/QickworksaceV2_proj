"""t1_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import T1GEParameters
from .program import T1GEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["T1GEParameters", "T1GEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "t1_ge",
    T1GEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="T1GE: independently maintained native GE protocol",
)

T1GEProgram.EXPERIMENT = experiment
