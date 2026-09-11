"""time_rabi_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import TimeRabiEFParameters
from .program import TimeRabiEFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["TimeRabiEFParameters", "TimeRabiEFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "time_rabi_ef",
    TimeRabiEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="TimeRabiEF: independently maintained native EF protocol",
)

TimeRabiEFProgram.EXPERIMENT = experiment
