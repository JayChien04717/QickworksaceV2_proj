"""time_rabi_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import TimeRabiGEParameters
from .program import TimeRabiGEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["TimeRabiGEParameters", "TimeRabiGEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "time_rabi_ge",
    TimeRabiGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="TimeRabiGE: independently maintained native GE protocol",
)

TimeRabiGEProgram.EXPERIMENT = experiment
