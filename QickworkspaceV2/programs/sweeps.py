from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Sweep:
    """An explicit axis bound to a native pulse parameter or tagged delay."""

    name: str
    start: float
    stop: float
    points: int
    unit: str = ""
    loop: str = "sweep"
    parameter: str = ""
    pulse: str | None = None
    tag: str | None = None

    def __post_init__(self):
        if isinstance(self.points, bool) or not isinstance(self.points, int) or self.points < 2:
            raise ValueError("Sweep needs at least two integer points")
        if not np.isfinite([self.start, self.stop]).all() or self.start == self.stop:
            raise ValueError("Sweep endpoints must be finite and distinct")
        if bool(self.pulse) == bool(self.tag):
            raise ValueError("Bind sweep to exactly one pulse or time tag")

    @property
    def values(self):
        return np.linspace(self.start, self.stop, self.points)

    def qick(self):
        from qick.asm_v2 import QickSweep1D

        return QickSweep1D(self.loop, self.start, self.stop)
