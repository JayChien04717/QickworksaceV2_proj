"""Common instrument interfaces and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


LimitMap = dict[str, tuple[float, float]]


@dataclass
class InstrumentSpec:
    """Registry entry stored by :class:`BaseInstrumentManager`."""

    name: str
    kind: str
    address: str
    driver: Any
    limits: LimitMap = field(default_factory=dict)
    notes: str = ""


class BaseInstrument:
    """Thin interface shared by instrument drivers."""

    KIND = "instrument"
    MODEL = "Generic Instrument"
    DEFAULT_LIMITS: LimitMap = {}

    def idn(self) -> str:
        """Return the instrument identification string.

        Returns
        -------
        str
            Result of the operation.
        """
        query = getattr(self, "query", None)
        if callable(query):
            return str(query("*IDN?"))
        return self.MODEL

    def get_limits(self) -> LimitMap:
        """Return limits.

        Returns
        -------
        LimitMap
            Result of the operation.
        """
        return dict(self.DEFAULT_LIMITS)

    def snapshot(self) -> dict[str, Any]:
        """Return the current instrument settings.

        Returns
        -------
        dict[str, Any]
            Result of the operation.
        """
        return {}


class SourceInstrument(BaseInstrument):
    """Base interface for instruments with an output state."""

    def on(self) -> None:
        """Enable the instrument output.

        Raises
        ------
        NotImplementedError
            If the operation cannot be completed.
        """
        raise NotImplementedError

    def off(self) -> None:
        """Disable the instrument output.

        Raises
        ------
        NotImplementedError
            If the operation cannot be completed.
        """
        raise NotImplementedError


class RFSourceInstrument(SourceInstrument):
    """Base interface for RF sources."""

    KIND = "rf_source"


class DCSourceInstrument(SourceInstrument):
    """Base interface for DC/flux-bias sources."""

    KIND = "dc_source"


def merge_limits(
    default_limits: Mapping[str, tuple[float, float]] | None,
    user_limits: Mapping[str, tuple[float, float]] | None,
) -> LimitMap:
    """Return driver defaults overridden by lab/user safety limits.

    Parameters
    ----------
    default_limits : Mapping[str, tuple[float, float]] | None
        Value for ``default_limits``.
    user_limits : Mapping[str, tuple[float, float]] | None
        Value for ``user_limits``.

    Returns
    -------
    LimitMap
        Result of the operation.
    """

    merged: LimitMap = dict(default_limits or {})
    merged.update(dict(user_limits or {}))
    return merged
