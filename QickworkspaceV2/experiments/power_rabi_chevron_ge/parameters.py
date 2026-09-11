"""Local defaults and allowed ranges for power_rabi_chevron_ge.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class PowerRabiChevronGEParameters(Parameters):
    # Repeated Rabi pulses per measurement; separate from averaging counts.
    iterations: int = Field(
        default=3,  # Used when this parameter is omitted.
        ge=1,  # Greater than or equal to 1 (inclusive minimum).
        le=101,  # Less than or equal to 101 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Starting GE drive amplitude in normalized QICK gain units, not dBm.
    start: float = Field(
        default=0.0,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
    )

    # Final GE drive amplitude in normalized QICK gain units, not dBm.
    stop: float = Field(
        default=0.6,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
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
