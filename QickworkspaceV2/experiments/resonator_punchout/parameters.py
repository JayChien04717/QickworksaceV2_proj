"""Local defaults for the FPGA gain-by-frequency punch-out map.

Catalog frequencies are offsets in MHz; native run_cfg sweeps use absolute MHz.
g_steps and f_steps set the two FPGA loop counts, not averaging counts.
"""

from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class PunchoutParameters(Parameters):
    # Frequency offsets from each target's configured resonator frequency (MHz).
    freq_start: float = Field(default=-10, description="Start readout-frequency offset (MHz)")
    freq_stop: float = Field(default=10, description="Stop readout-frequency offset (MHz)")

    # Frequency columns in each gain row, including both endpoints.
    f_steps: int = Field(
        default=101,  # Default frequency point count.
        ge=2,  # At least two points, inclusive.
        le=10001,  # Maximum point count, inclusive.
        strict=True,  # Require an integer.
    )

    # Absolute resonator amplitudes in normalized QICK gain units, not multipliers.
    gain_start: float = Field(
        default=0.02,
        ge=0,  # Zero is allowed.
        le=1,  # Normalized amplitude cannot exceed one.
    )
    gain_stop: float = Field(
        default=0.2,
        gt=0,  # Must be strictly positive.
        le=1,  # Inclusive maximum; configured hardware limits still apply.
    )

    # Gain rows in the 2D map, including both endpoints.
    g_steps: int = Field(
        default=11,
        ge=2,  # Inclusive minimum.
        le=1001,  # Inclusive maximum.
        strict=True,  # Require an integer.
    )

    @model_validator(mode="after")
    def ordered_ranges(self):
        if self.freq_start >= self.freq_stop or self.gain_start >= self.gain_stop:
            raise ValueError("Frequency and gain starts must be below their stops")
        return self
