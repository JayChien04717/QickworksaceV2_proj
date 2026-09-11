"""QICK-native multi-qubit measurement workspace. Hardware imports are optional."""

__version__ = "2.0.0"

from QickworkspaceV2.device import Device, DeviceConfig, HardwareConfig, RunDefaults
from QickworkspaceV2.data.models import ExperimentData, TraceData, FitResult, QualityFlag
from QickworkspaceV2.programs.sweeps import Sweep
from QickworkspaceV2.experiments.base import (
    Parameters,
    ProgramPlan,
    BuildContext,
    ExperimentSpec,
    ExperimentRegistry,
)
from QickworkspaceV2.backends import QICKBackend
from QickworkspaceV2.calibration import (
    CalibrationStore,
    CalibrationProposal,
    CalibrationGraph,
    CalibrationNode,
)
from QickworkspaceV2.runtime.session import Session
from QickworkspaceV2.runtime.instrument_scan import InstrumentAxis


def __getattr__(name):
    if name == "QickSweep1D":
        from qick.asm_v2 import QickSweep1D
        return QickSweep1D
    if name == "BaseProgram":
        from QickworkspaceV2.programs.base import BaseProgram

        return BaseProgram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Session",
    "Device",
    "DeviceConfig",
    "HardwareConfig",
    "RunDefaults",
    "BaseProgram",
    "InstrumentAxis",
    "ExperimentData",
    "TraceData",
    "FitResult",
    "QualityFlag",
    "Parameters",
    "ProgramPlan",
    "BuildContext",
    "ExperimentSpec",
    "ExperimentRegistry",
    "Sweep",
    "QICKBackend",
    "CalibrationStore",
    "CalibrationProposal",
    "CalibrationGraph",
    "CalibrationNode",
]

from QickworkspaceV2.device.editable import ExperimentConfig, RunConfig

__all__ += ["ExperimentConfig", "RunConfig"]

from QickworkspaceV2.runtime.measurement import Measurement

__all__ += ["Measurement"]

from QickworkspaceV2.notebook import NotebookLab
from numpy import linspace

__all__ += ["NotebookLab", "QickSweep1D", "linspace"]
