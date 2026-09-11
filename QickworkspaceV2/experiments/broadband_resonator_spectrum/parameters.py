"""Local defaults and allowed ranges for broadband_resonator_spectrum.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class DetectionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Minimum separation between magnitude-dip candidates in samples, not MHz.
    min_distance_points: int = Field(
        default=6,  # Used when this parameter is omitted.
        ge=1,  # Greater than or equal to 1 (inclusive minimum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Prominence threshold multiplier of the smoothed negative-magnitude standard deviation.
    prominence_sigma: float = Field(
        default=0.08,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Width penalty exponent for candidate ranking; 0 ranks by prominence alone.
    width_penalty: float = Field(
        default=0.5,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Allow phase-derivative candidates to help locate resonators.
    use_phase_reference: bool = True  # Default value.

    # Maximum frequency separation for pairing magnitude and phase candidates (Hz); 15e6 = 15 MHz.
    phase_snap_hz: float = Field(
        default=15e6,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )


class BroadbandParameters(Parameters):
    # Starting offset from the configured resonator frequency (MHz), not an absolute frequency.
    start: float = Field(
        default=-200,  # Used when this parameter is omitted.
        description="Start offset from calibrated readout frequency (MHz)",  # Parameter help shown in the catalog/UI.
    )

    # Final offset from the configured resonator frequency (MHz), not an absolute frequency.
    stop: float = Field(
        default=200,  # Used when this parameter is omitted.
        description="Stop offset from calibrated readout frequency (MHz)",  # Parameter help shown in the catalog/UI.
    )

    # Number of sweep points, including both endpoints; separate from reps and py_avg.
    points: int = Field(
        default=401,  # Used when this parameter is omitted.
        ge=8,  # Greater than or equal to 8 (inclusive minimum).
        le=10001,  # Less than or equal to 10001 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Readout gain multiplier (unitless): 1 keeps the configured gain; 0.5 halves it.
    gain_scale: float = Field(
        default=1,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
        le=5,  # Less than or equal to 5 (inclusive maximum).
    )

    # Requested resonator count; analysis fails if too few candidates are found.
    count: int = Field(
        default=4,  # Used when this parameter is omitted.
        ge=1,  # Greater than or equal to 1 (inclusive minimum).
        le=100,  # Less than or equal to 100 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Local dip-detection settings; see DetectionOptions above for each field.
    detection_options: DetectionOptions = Field(
        default_factory=DetectionOptions,  # Create independent default settings for each instance.
    )

    # Final plot vertical axis: abs for magnitude, db for decibels, phase for phase.
    y_mode: Literal["abs", "db", "phase"] = "abs"  # Default value.

    # Each sweep start must be below its stop; individual field bounds do not check this.
    @model_validator(mode="after")
    def ordered_range(self):
        if self.start >= self.stop:
            raise ValueError("start must be below stop")
        return self
