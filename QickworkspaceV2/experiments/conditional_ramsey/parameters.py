"""Local defaults and allowed ranges for conditional_ramsey.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field
from QickworkspaceV2.experiments.base import Parameters


class ConditionalRamseyParameters(Parameters):
    # Two fixed qubit IDs separated by a comma. Order: control,target.
    target: str = "Q1,Q2"  # Default value.

    # Starting Ramsey free-evolution wait (us).
    start: float = Field(
        default=0.0,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Final Ramsey free-evolution wait (us).
    stop: float = Field(
        default=30.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Number of sweep points, including both endpoints; separate from reps and py_avg.
    points: int = Field(
        default=81,  # Used when this parameter is omitted.
        ge=8,  # Greater than or equal to 8 (inclusive minimum).
        le=10001,  # Less than or equal to 10001 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Analysis-pulse phase ramp (MHz): phase = 360 * detuning_mhz * wait_us.
    detuning_mhz: float = 0.2  # Default value.

    # Relaxation wait between state preparation/measurement sequences (us).
    reset_wait_us: float = Field(
        default=200.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )
