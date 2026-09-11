"""qubit_spec_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import QubitSpecEFParameters
from .program import QubitSpecEFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["QubitSpecEFParameters", "QubitSpecEFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "qubit_spec_ef",
    QubitSpecEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="QubitSpecEF: independently maintained native EF protocol",
)

QubitSpecEFProgram.EXPERIMENT = experiment
