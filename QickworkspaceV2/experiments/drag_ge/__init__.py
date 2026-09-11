"""drag_ge: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import DragGEParameters
from .program import DragGEProgram, build
from .analysis import analyze, plot, updates

__all__ = ["DragGEParameters", "DragGEProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "drag_ge",
    DragGEParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="GE DRAG alpha point with repeated X/-X and reference states",
)

DragGEProgram.EXPERIMENT = experiment
