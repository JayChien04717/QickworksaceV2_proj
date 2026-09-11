"""power_rabi_chevron_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import PowerRabiChevronEFParameters
from .program import PowerRabiChevronEFProgram, build
from .analysis import analyze, plot, updates

__all__ = [
    "PowerRabiChevronEFParameters",
    "PowerRabiChevronEFProgram",
    "build",
    "analyze",
    "plot",
    "experiment",
]

experiment = ExperimentSpec(
    "power_rabi_chevron_ef",
    PowerRabiChevronEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="PowerRabiChevronEF: independently maintained native EF protocol",
)

PowerRabiChevronEFProgram.EXPERIMENT = experiment
