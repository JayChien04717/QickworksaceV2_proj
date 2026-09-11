"""resonator_flux: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResonatorFluxParameters
from .program import ResonatorFluxProgram, build
from .analysis import analyze, plot, updates

__all__ = [
    "ResonatorFluxParameters",
    "ResonatorFluxProgram",
    "build",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "resonator_flux",
    ResonatorFluxParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Readout spectroscopy under a timed QICK bias pulse",
)

ResonatorFluxProgram.EXPERIMENT = experiment
