"""drag_ef: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import DragEFParameters
from .program import DragEFProgram, build
from .analysis import analyze, plot, updates

__all__ = ["DragEFParameters", "DragEFProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "drag_ef",
    DragEFParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="EF DRAG alpha point with repeated X/-X and reference states",
)

DragEFProgram.EXPERIMENT = experiment
