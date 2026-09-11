"""Small data contract shared by experiment acquisition paths.

The experiment lifecycle itself lives in :mod:`base_experiment`.  Keeping it
in one place makes the order of program construction, sweep resolution,
acquisition, fitting, and analysis explicit.  This module only defines the
payload returned by custom acquisition implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .experiment_data import QualityFlag


@dataclass(frozen=True)
class SweepAxis:
    """Declarative QICK sweep coordinate used by ordinary experiments."""

    kind: str
    target: str
    parameter: str

    def __post_init__(self):
        if self.kind not in {"pulse", "time"}:
            raise ValueError("SweepAxis kind must be 'pulse' or 'time'")
        if not self.target or not self.parameter:
            raise ValueError("SweepAxis target and parameter cannot be empty")

    @classmethod
    def pulse(cls, pulse_name: str, parameter: str):
        """Read an axis through ``program.get_pulse_param``."""
        return cls("pulse", pulse_name, parameter)

    @classmethod
    def time(cls, tag: str, parameter: str = "t"):
        """Read an axis through ``program.get_time_param``."""
        return cls("time", tag, parameter)

    def extract(self, program):
        getter = (
            program.get_pulse_param
            if self.kind == "pulse"
            else program.get_time_param
        )
        return getter(self.target, self.parameter, as_array=True)


@dataclass
class AcquisitionResult:
    """Raw hardware output before it is promoted to ``ExperimentData``."""

    raw_iq: Any = None
    interrupted: bool = False
    avg_count: int = 0
    quality: QualityFlag = QualityFlag.NO_INFORMATION
    quality_message: str = ""
    fit_params: Any = None
    fit_errors: Any = None
    fit_result: dict = field(default_factory=dict)
    scalar_result: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)
    raw_data: dict = field(default_factory=dict)
    analysis_data: dict = field(default_factory=dict)
    dataset_dims: dict = field(default_factory=dict)


def infer_iq_dims(shape, *, x_axis=None, y_axis=None) -> list[str]:
    """Infer dimension names without claiming fewer dimensions than the data."""
    rank = len(shape)
    if rank == 0:
        return []
    if y_axis is not None and rank >= 2:
        trailing = ["y", "x"]
    elif x_axis is not None:
        trailing = ["x"]
    else:
        trailing = []
    leading_count = rank - len(trailing)
    leading = [
        "readout" if leading_count == 1 else f"dim_{index}"
        for index in range(leading_count)
    ]
    return [*leading, *trailing]


__all__ = ["SweepAxis", "AcquisitionResult", "infer_iq_dims"]
