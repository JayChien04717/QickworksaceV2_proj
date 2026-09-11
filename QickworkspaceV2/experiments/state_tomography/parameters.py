"""Local defaults and allowed ranges for state_tomography.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from pydantic import Field
from QickworkspaceV2.experiments.base import Parameters


class ResetParameters(Parameters):
    # Relaxation wait between state preparation/measurement sequences (us).
    reset_wait_us: float = Field(
        default=200.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )


class TomographyParameters(ResetParameters):
    # Initial tomography state: ground, excited or plus; customize sequences in program.py.
    preparation: str = Field(
        default="plus",  # Used when this parameter is omitted.
        description="ground, excited or plus; custom state preparation belongs in a custom program",  # Parameter help shown in the catalog/UI.
    )
