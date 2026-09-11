"""dispersive: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import DispersiveParameters
from .program import DispersiveProgram, build
from .analysis import analyze, plot, updates

__all__ = ["DispersiveParameters", "DispersiveProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "dispersive",
    DispersiveParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Paired g/e frequency sweep and readout SNR",
)

DispersiveProgram.EXPERIMENT = experiment
