"""resonator_spec_f: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResonatorSpecFParameters
from .program import ResonatorSpecFProgram, build_resonator
from .analysis import analyze, plot, updates

__all__ = [
    "ResonatorSpecFParameters",
    "ResonatorSpecFProgram",
    "build_resonator",
    "analyze",
    "plot",
    "experiment",
]

experiment = ExperimentSpec(
    "resonator_spec_f",
    ResonatorSpecFParameters,
    build_resonator,
    analyze,
    plot=plot,
    updates=updates,
    description="Readout resonance after preparing |f>; direct and mux",
)

ResonatorSpecFProgram.EXPERIMENT = experiment
