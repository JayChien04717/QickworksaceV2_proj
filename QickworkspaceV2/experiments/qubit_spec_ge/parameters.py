"""Local defaults and allowed ranges for qubit_spec_ge.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class QubitSpecGEParameters(Parameters):
    # Starting offset from the configured qubit GE frequency (MHz), not an absolute frequency.
    start: float = -20.0  # Default value.

    # Final offset from the configured qubit GE frequency (MHz), not an absolute frequency.
    stop: float = 20.0  # Default value.

    # Qubit drive amplitude in normalized QICK gain units, not dBm.
    gain: float = Field(
        default=0.05,  # Used when this parameter is omitted.
        ge=-1,  # Greater than or equal to -1 (inclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
    )

    # Flat portion of the qubit spectroscopy drive pulse (us).
    length_us: float = Field(
        default=2.0,  # Used when this parameter is omitted.
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
