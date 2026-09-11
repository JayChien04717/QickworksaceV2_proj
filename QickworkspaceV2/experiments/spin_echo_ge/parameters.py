"""Local defaults and allowed ranges for spin_echo_ge.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class SpinEchoGEParameters(Parameters):
    # Starting total free-evolution time (us); each echo interval uses half this value.
    start: float = Field(
        default=0.0,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Final total free-evolution time (us); each echo interval uses half this value.
    stop: float = Field(
        default=100.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Number of sweep points, including both endpoints; separate from reps and py_avg.
    points: int = Field(
        default=81,  # Used when this parameter is omitted.
        ge=8,  # Greater than or equal to 8 (inclusive minimum).
        le=10001,  # Less than or equal to 10001 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Each sweep start must be below its stop; individual field bounds do not check this.
    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self
