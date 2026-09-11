"""coupler_chevron: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ChevronParameters
from .program import ChevronProgram, build_chevron
from .analysis import analyze, plot, updates

__all__ = ["ChevronParameters", "ChevronProgram", "build_chevron", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "coupler_chevron",
    ChevronParameters,
    build_chevron,
    analyze,
    plot=plot,
    updates=updates,
    description="Two-dimensional native coupler interaction scan",
)

ChevronProgram.EXPERIMENT = experiment
