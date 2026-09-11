"""resonator_spec: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResonatorParameters
from .program import ResonatorSpecProgram, build_resonator
from .analysis import analyze, plot, updates

__all__ = [
    "ResonatorParameters",
    "ResonatorSpecProgram",
    "build_resonator",
    "analyze",
    "plot",
    "updates",
    "experiment",
]

experiment = ExperimentSpec(
    "resonator_spec",
    ResonatorParameters,
    build_resonator,
    analyze,
    plot=plot,
    updates=updates,
    description="Readout resonance; host sweep supports direct and mux",
)

ResonatorSpecProgram.EXPERIMENT = experiment
