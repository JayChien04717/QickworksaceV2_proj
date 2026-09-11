"""power_rabi_chevron_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import PowerRabiChevronGEParameters
from .program import PowerRabiChevronGEProgram, build
from .analysis import analyze, plot, updates

__all__ = [
    "PowerRabiChevronGEParameters",
    "PowerRabiChevronGEProgram",
    "build",
    "analyze",
    "plot",
    "experiment",
]

experiment = ExperimentSpec(
    "power_rabi_chevron_ge",
    PowerRabiChevronGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="PowerRabiChevronGE: independently maintained native GE protocol",
)

PowerRabiChevronGEProgram.EXPERIMENT = experiment
