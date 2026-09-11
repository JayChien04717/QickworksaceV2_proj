"""Local defaults and allowed ranges for time_of_flight.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field, model_validator
from QickworkspaceV2.experiments.base import Parameters


class TimeOfFlightParameters(Parameters):
    """Targets for the decimated readout diagnostic."""

    # Prepare e with a GE pi pulse; cannot be enabled together with check_f.
    check_e: bool = Field(
        default=False,  # Used when this parameter is omitted.
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Prepare f with GE and EF pi pulses; both checks False selects g.
    check_f: bool = Field(
        default=False,  # Used when this parameter is omitted.
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Extra wait between state preparation and readout (us).
    prepare_delay_us: float = Field(
        default=0.05,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # Absolute TOF envelope threshold (ADC units); None leaves it unset, separate from ro_threshold.
    tof_threshold: float | None = Field(
        default=None,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
    )

    # check_e and check_f cannot both be True.
    @model_validator(mode="after")
    def one_prepared_state(self):
        if self.check_e and self.check_f:
            raise ValueError("Choose check_e or check_f, not both")
        return self
