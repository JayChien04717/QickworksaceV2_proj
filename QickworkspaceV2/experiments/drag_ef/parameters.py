"""Local defaults and allowed ranges for drag_ef.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from pydantic import Field, model_validator
from QickworkspaceV2 import Parameters


class DragEFParameters(Parameters):
    # Dimensionless DRAG coefficient controlling the quadrature derivative correction.
    alpha: float = 0.0  # Default value.

    # Signed detuning for QICK add_DRAG (MHz); required and must be nonzero.
    delta_mhz: float = Field(
        description="Signed DRAG detuning used by QICK add_DRAG; specify the measured value",  # Parameter help shown in the catalog/UI.
    )

    # Number of X / -X pulse pairs per measurement.
    iterations: int = Field(
        default=5,  # Used when this parameter is omitted.
        ge=1,  # Greater than or equal to 1 (inclusive minimum).
        le=101,  # Less than or equal to 101 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Relaxation wait between state preparation/measurement sequences (us).
    reset_wait_us: float = Field(
        default=200.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # DRAG detuning may have either sign, but must be nonzero.
    @model_validator(mode="after")
    def nonzero_delta(self):
        if self.delta_mhz == 0:
            raise ValueError("DRAG delta_mhz must be nonzero")
        return self
