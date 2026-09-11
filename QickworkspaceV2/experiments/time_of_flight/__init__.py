"""time_of_flight: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import TimeOfFlightParameters
from .program import TimeOfFlightProgram, build_tof
from .analysis import analyze, plot, updates

__all__ = [
    "TimeOfFlightParameters",
    "TimeOfFlightProgram",
    "build_tof",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "time_of_flight",
    TimeOfFlightParameters,
    build_tof,
    analyze,
    plot=plot,
    updates=updates,
    description="Raw decimated I/Q with optional e/f preparation and envelope diagnostics",
)

TimeOfFlightProgram.EXPERIMENT = experiment
