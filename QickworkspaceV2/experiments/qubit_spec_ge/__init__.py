"""qubit_spec_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import QubitSpecGEParameters
from .program import QubitSpecGEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["QubitSpecGEParameters", "QubitSpecGEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "qubit_spec_ge",
    QubitSpecGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="QubitSpecGE: independently maintained native GE protocol",
)

QubitSpecGEProgram.EXPERIMENT = experiment
