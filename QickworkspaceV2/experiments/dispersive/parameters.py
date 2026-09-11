"""Local defaults and allowed ranges for dispersive.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from pydantic import Field
from QickworkspaceV2 import Parameters


class DispersiveParameters(Parameters):
    # Starting offset from the configured resonator frequency (MHz), not an absolute frequency.
    start: float = -5  # Default value.

    # Final offset from the configured resonator frequency (MHz), not an absolute frequency.
    stop: float = 5  # Default value.

    # Number of sweep points, including both endpoints; separate from reps and py_avg.
    points: int = Field(
        default=51,  # Used when this parameter is omitted.
        ge=8,  # Greater than or equal to 8 (inclusive minimum).
        le=1001,  # Less than or equal to 1001 (inclusive maximum).
    )

    # Relaxation wait between state preparation/measurement sequences (us).
    reset_wait_us: float = Field(
        default=200,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )
