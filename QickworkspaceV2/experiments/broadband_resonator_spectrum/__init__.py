"""Independent broadband resonator spectrum experiment."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import BroadbandParameters, DetectionOptions
from .program import BroadbandResonatorSpecProgram, build
from .analysis import analyze, plot, updates

experiment = ExperimentSpec(
    "broadband_resonator_spectrum",
    BroadbandParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Broadband one-tone spectrum with phase-referenced multi-resonator frequency detection",
)
BroadbandResonatorSpecProgram.EXPERIMENT = experiment

__all__ = [
    "BroadbandParameters",
    "DetectionOptions",
    "BroadbandResonatorSpecProgram",
    "build",
    "analyze",
    "plot",
    "updates",
    "experiment",
]
