"""Local defaults and allowed ranges for single_shot.

Each field documents its purpose and units. Field lists defaults and validation
limits separately: ge/le include the boundary; gt/lt exclude it.
Here ge means a numeric comparison, not the qubit GE transition.
See program.py for parameter-to-run_cfg mapping and pulse sequences;
see analysis.py for analysis, final plots and proposed calibration updates.
"""

from __future__ import annotations
from typing import Literal
from pydantic import Field
from QickworkspaceV2.experiments.base import Parameters


class ResetParameters(Parameters):
    # Relaxation wait between state preparation/measurement sequences (us).
    reset_wait_us: float = Field(
        default=200.0,  # Used when this parameter is omitted.
        gt=0,  # Strictly greater than 0 (exclusive minimum).
    )

    # Classification axis: auto chooses a rotation; I or Q selects that component.
    classifier_axis: Literal["auto", "I", "Q"] = "auto"  # Default value.
