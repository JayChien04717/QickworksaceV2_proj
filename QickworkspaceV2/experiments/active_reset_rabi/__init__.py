"""active_reset_rabi: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ActiveResetParameters
from .program import ActiveResetRabiProgram, build
from .analysis import analyze, plot, updates

__all__ = [
    "ActiveResetParameters",
    "ActiveResetRabiProgram",
    "build",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "active_reset_rabi",
    ActiveResetParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Native I/Q threshold feedback; paired pre/post-reset shots",
)

ActiveResetRabiProgram.EXPERIMENT = experiment
