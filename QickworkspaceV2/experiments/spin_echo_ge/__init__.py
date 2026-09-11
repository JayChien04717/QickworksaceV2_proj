"""spin_echo_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import SpinEchoGEParameters
from .program import SpinEchoGEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["SpinEchoGEParameters", "SpinEchoGEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "spin_echo_ge",
    SpinEchoGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="SpinEchoGE: independently maintained native GE protocol",
)

SpinEchoGEProgram.EXPERIMENT = experiment
