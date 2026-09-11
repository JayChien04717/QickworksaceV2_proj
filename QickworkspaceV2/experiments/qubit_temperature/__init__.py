"""qubit_temperature: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import TemperatureParameters
from .program import QubitTemperatureProgram, build
from .analysis import analyze, plot, updates

__all__ = [
    "TemperatureParameters",
    "QubitTemperatureProgram",
    "build",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "qubit_temperature",
    TemperatureParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Paired EF Rabi estimate of thermal population and effective temperature",
)

QubitTemperatureProgram.EXPERIMENT = experiment
