"""resonator_spec_e: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import ResonatorSpecEParameters
from .program import ResonatorSpecEProgram, build_resonator
from .analysis import analyze, plot, updates

__all__ = [
    "ResonatorSpecEParameters",
    "ResonatorSpecEProgram",
    "build_resonator",
    "analyze",
    "plot",
    "experiment",
]

experiment = ExperimentSpec(
    "resonator_spec_e",
    ResonatorSpecEParameters,
    build_resonator,
    analyze,
    plot=plot,
    updates=updates,
    description="Readout resonance after preparing |e>; direct and mux",
)

ResonatorSpecEProgram.EXPERIMENT = experiment
