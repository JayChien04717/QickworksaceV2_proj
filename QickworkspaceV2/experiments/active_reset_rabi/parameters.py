"""Local defaults and allowed ranges for active_reset_rabi.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from typing import Literal
from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class ActiveResetParameters(Parameters):
    # Starting GE drive amplitude in normalized QICK gain units, not dBm.
    start: float = Field(
        default=0,  # Used when this parameter is omitted.
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

    # Reset policy: never skips the pulse, conditional uses feedback, always applies it.
    reset_mode: Literal["never", "conditional", "always"] = "conditional"  # Default value.

    # Raw I or Q component used by hardware feedback; match the calibrated threshold.
    reset_component: Literal["I", "Q"] = "I"  # Default value.

    # Excited-state comparison: >= treats readings at or above the threshold as excited.
    reset_excited_if: Literal[">=", "<"] = ">="  # Default value.

    # Extra timing margin when waiting for readout and generators to finish (us).
    read_wait_us: float = Field(
        default=0.2,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Feedback resync timing margin before applying the reset pulse (us).
    feedback_slack_us: float = Field(
        default=0.05,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Wait after the reset decision/pulse before the second readout (us).
    reset_post_delay_us: float = Field(
        default=0.05,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Each sweep start must be below its stop; individual field bounds do not check this.
    @model_validator(mode="after")
    def ordered(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self
