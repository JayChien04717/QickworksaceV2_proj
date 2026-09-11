"""Local defaults and allowed ranges for resonator_flux.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from pydantic import Field, model_validator
from QickworkspaceV2 import Parameters


class ResonatorFluxParameters(Parameters):
    # Logical flux/bias generator ID in the hardware settings, not a channel index.
    bias_port: str = Field(
        description="Logical generator ID connected to the flux/bias line",  # Parameter help shown in the catalog/UI.
    )

    # Flux/bias pulse amplitude in normalized QICK gain units, not volts.
    bias_gain: float = Field(
        default=0.0,  # Used when this parameter is omitted.
        ge=-1,  # Greater than or equal to -1 (inclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
    )

    # Wait after starting the bias pulse before beginning readout (us).
    settle_us: float = Field(
        default=0.1,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Duration of the bias pulse (us).
    bias_length_us: float = Field(
        default=10.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Starting offset from the configured resonator frequency (MHz), not an absolute frequency.
    start: float = -3.0  # Default value.

    # Final offset from the configured resonator frequency (MHz), not an absolute frequency.
    stop: float = 3.0  # Default value.

    # Number of sweep points, including both endpoints; separate from reps and py_avg.
    points: int = Field(
        default=41,  # Used when this parameter is omitted.
        ge=8,  # Greater than or equal to 8 (inclusive minimum).
        le=10001,  # Less than or equal to 10001 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Each sweep start must be below its stop; individual field bounds do not check this.
    @model_validator(mode="after")
    def ordered(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self
