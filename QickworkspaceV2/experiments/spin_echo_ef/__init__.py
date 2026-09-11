"""spin_echo_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import SpinEchoEFParameters
from .program import SpinEchoEFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["SpinEchoEFParameters", "SpinEchoEFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "spin_echo_ef",
    SpinEchoEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="SpinEchoEF: independently maintained native EF protocol",
)

SpinEchoEFProgram.EXPERIMENT = experiment
