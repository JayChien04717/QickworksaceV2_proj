"""conditional_ramsey: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ConditionalRamseyParameters
from .program import ConditionalRamseyProgram, build_conditional
from .analysis import analyze_conditional, plot, updates

__all__ = [
    "ConditionalRamseyParameters",
    "ConditionalRamseyProgram",
    "build_conditional",
    "analyze_conditional",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "conditional_ramsey",
    ConditionalRamseyParameters,
    build_conditional,
    analyze_conditional,
    plot=plot,
    updates=updates,
    description="Two-qubit conditional Ramsey: control,target",
)

ConditionalRamseyProgram.EXPERIMENT = experiment
