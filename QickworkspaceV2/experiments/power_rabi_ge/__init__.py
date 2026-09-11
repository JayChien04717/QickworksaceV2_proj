"""power_rabi_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import PowerRabiGEParameters
from .program import PowerRabiGEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["PowerRabiGEParameters", "PowerRabiGEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "power_rabi_ge",
    PowerRabiGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="PowerRabiGE: independently maintained native GE protocol",
)

PowerRabiGEProgram.EXPERIMENT = experiment
