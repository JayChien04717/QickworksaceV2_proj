"""twpa_probe: native program, local parameters, analysis and catalog entry."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .parameters import TWPAProbeParameters
from .program import TWPAProbeProgram, build
from .analysis import analyze, plot, updates

__all__ = ["TWPAProbeParameters", "TWPAProbeProgram", "build", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "twpa_probe",
    TWPAProbeParameters,
    build,
    analyze,
    plot=plot,
    updates=updates,
    description="Broadband complex transmission; compare matched pump-on/off reference runs",
)

TWPAProbeProgram.EXPERIMENT = experiment
