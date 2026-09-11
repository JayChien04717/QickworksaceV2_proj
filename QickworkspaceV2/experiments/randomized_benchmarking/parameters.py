"""Local defaults and allowed ranges for randomized_benchmarking.

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


class RBParameters(ResetParameters):
    # Comma-separated Clifford counts; each count defines one RB sequence depth.
    depths: str = Field(
        default="0,1,2,4,8,12,16,24",  # Used when this parameter is omitted.
        description="Comma-separated nonnegative Clifford counts",  # Parameter help shown in the catalog/UI.
    )

    # Random sequences generated at each RB depth; separate from hardware averages.
    sequences_per_depth: int = Field(
        default=2,  # Used when this parameter is omitted.
        ge=2,  # Greater than or equal to 2 (inclusive minimum).
        le=20,  # Less than or equal to 20 (inclusive maximum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Random seed; identical seeds and parameters reproduce the same RB sequences.
    seed: int = Field(
        default=42,  # Used when this parameter is omitted.
        ge=0,  # Greater than or equal to 0 (inclusive minimum).
        strict=True,  # Require the declared type without automatic conversion.
    )

    # Test gate inserted between Cliffords; none selects reference RB.
    interleaved: Literal["none", "x", "y", "halfx", "halfy", "mhalfx", "mhalfy"] = "none"  # Default value.
