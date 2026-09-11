"""Local defaults and allowed ranges for ckp.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from pydantic import Field
from QickworkspaceV2 import Parameters


class CKPParameters(Parameters):
    # Start frequency offset from the configured resonator frequency (MHz).
    readout_start: float = -10  # Default value.

    # Stop frequency offset from the configured resonator frequency (MHz).
    readout_stop: float = 10  # Default value.

    # Start frequency offset from the configured qubit GE frequency (MHz).
    qubit_start: float = -20  # Default value.

    # Stop frequency offset from the configured qubit GE frequency (MHz).
    qubit_stop: float = 20  # Default value.

    # Readout-frequency point count; combines with qubit_points to form a 2D sweep.
    readout_points: int = Field(
        default=21,  # Used when this parameter is omitted.
        ge=2,  # Greater than or equal to 2 (inclusive minimum).
        le=501,  # Less than or equal to 501 (inclusive maximum).
    )

    # Qubit-frequency point count; separate from averaging counts.
    qubit_points: int = Field(
        default=41,  # Used when this parameter is omitted.
        ge=2,  # Greater than or equal to 2 (inclusive minimum).
        le=501,  # Less than or equal to 501 (inclusive maximum).
    )

    # CKP qubit drive amplitude in normalized QICK gain units.
    drive_gain: float = Field(
        default=0.05,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
        le=1,  # Less than or equal to 1 (inclusive maximum).
    )

    # Flat portion of the CKP qubit drive pulse (us).
    drive_length_us: float = Field(
        default=5,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )
