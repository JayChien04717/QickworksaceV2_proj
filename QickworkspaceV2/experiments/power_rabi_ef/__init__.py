"""power_rabi_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import PowerRabiEFParameters
from .program import PowerRabiEFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["PowerRabiEFParameters", "PowerRabiEFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "power_rabi_ef",
    PowerRabiEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="PowerRabiEF: independently maintained native EF protocol",
)

PowerRabiEFProgram.EXPERIMENT = experiment
