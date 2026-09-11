"""ckp: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import CKPParameters
from .program import CKPProgram, build
from .analysis import analyze, plot, updates

__all__ = ["CKPParameters", "CKPProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "ckp", CKPParameters, build, analyze, plot=plot, description="Two-dimensional readout/qubit frequency map"
)

CKPProgram.EXPERIMENT = experiment
