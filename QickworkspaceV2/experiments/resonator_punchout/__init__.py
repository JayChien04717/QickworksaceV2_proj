"""Public program, local parameters and catalog entry for FPGA punch-out."""

from QickworkspaceV2.experiments.base import ExperimentSpec
from .program import PunchoutProgram, build_punchout
from .parameters import PunchoutParameters
from .analysis import analyze, plot, updates

__all__ = ["PunchoutProgram", "PunchoutParameters", "build_punchout", "analyze", "plot", "updates", "experiment"]

experiment = ExperimentSpec(
    "resonator_punchout",
    PunchoutParameters,
    build_punchout,
    analyze,
    plot=plot,
    updates=updates,
    description="Readout punch-out: FPGA gain outer loop and frequency inner loop; direct tProc readout",
)

PunchoutProgram.EXPERIMENT = experiment
