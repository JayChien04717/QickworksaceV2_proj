"""allxy: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResetParameters
from .program import ALLXY, ALLXY_IDEAL, AllXYProgram, build_allxy
from .analysis import analyze_allxy, plot, updates

__all__ = [
    "ALLXY",
    "ALLXY_IDEAL",
    "ResetParameters",
    "AllXYProgram",
    "build_allxy",
    "analyze_allxy",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "allxy",
    ResetParameters,
    build_allxy,
    analyze_allxy,
    plot=plot,
    updates=updates,
    description="allxy: independent native protocol",
)

AllXYProgram.EXPERIMENT = experiment
