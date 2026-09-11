"""Local defaults and allowed ranges for coupler_chevron.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class ChevronParameters(Parameters):
    # Two fixed qubit IDs separated by a comma.
    target: str = "Q1,Q2"  # Default value.

    # Configured coupler ID selecting its connection and pulse settings.
    coupler: str = "C12"  # Default value.

    # Starting coupler amplitude in normalized QICK gain units.
    gain_start: float = Field(
        default=0.01,  # Used when this parameter is omitted.
        ge=-1,  # Greater than or equal to -1 (inclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
    )

    # Final coupler amplitude in normalized QICK gain units.
    gain_stop: float = Field(
        default=0.2,  # Used when this parameter is omitted.
        ge=-1,  # Greater than or equal to -1 (inclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
    )

    # Number of points along the coupler gain axis.
    gain_points: int = Field(
        default=21,  # Used when this parameter is omitted.
        ge=2,  # Greater than or equal to 2 (inclusive minimum).
        le=501,  # Less than or equal to 501 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Starting coupler interaction pulse length (us).
    length_start_us: float = Field(
        default=0.02,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Final coupler interaction pulse length (us).
    length_stop_us: float = Field(
        default=2.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Number of points along the coupler pulse-length axis.
    length_points: int = Field(
        default=81,  # Used when this parameter is omitted.
        ge=2,  # Greater than or equal to 2 (inclusive minimum).
        le=1001,  # Less than or equal to 1001 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Each sweep start must be below its stop; individual field bounds do not check this.
    @model_validator(mode="after")
    def ranges(self):
        if self.gain_start >= self.gain_stop or self.length_start_us >= self.length_stop_us:
            raise ValueError("Chevron starts must be below stops")
        return self
