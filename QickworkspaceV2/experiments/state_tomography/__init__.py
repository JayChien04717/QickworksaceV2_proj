"""state_tomography: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResetParameters, TomographyParameters
from .program import StateTomographyProgram, build_tomography
from .analysis import analyze_tomography, plot, updates

__all__ = [
    "ResetParameters",
    "TomographyParameters",
    "StateTomographyProgram",
    "build_tomography",
    "analyze_tomography",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "state_tomography",
    TomographyParameters,
    build_tomography,
    analyze_tomography,
    plot=plot,
    updates=updates,
    description="state tomography: independent native protocol",
)

StateTomographyProgram.EXPERIMENT = experiment
