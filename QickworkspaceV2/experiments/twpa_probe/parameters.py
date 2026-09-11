"""Local defaults and allowed ranges for twpa_probe.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from pydantic import Field, model_validator
from QickworkspaceV2 import Parameters


class TWPAProbeParameters(Parameters):
    # Absolute starting frequency of the TWPA probe sweep (MHz).
    start_mhz: float = 4000.0  # Default value.

    # Absolute final frequency of the TWPA probe sweep (MHz).
    stop_mhz: float = 8000.0  # Default value.

    # Number of sweep points, including both endpoints; separate from reps and py_avg.
    points: int = Field(
        default=101,  # Used when this parameter is omitted.
        ge=2,  # Greater than or equal to 2 (inclusive minimum).
        le=10001,  # Less than or equal to 10001 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Readout gain multiplier (unitless): 1 keeps the configured gain; 0.5 halves it.
    gain_scale: float = Field(
        default=0.1,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
        le=5,  # Less than or equal to 5 (inclusive maximum).
    )

    # Each sweep start must be below its stop; individual field bounds do not check this.
    @model_validator(mode="after")
    def ordered(self):
        if self.start_mhz >= self.stop_mhz:
            raise ValueError("start_mhz must be below stop_mhz")
        return self
