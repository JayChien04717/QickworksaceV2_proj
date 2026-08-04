"""
Calibration store: timestamped parameter persistence with staleness detection.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        """Return the default result.

        Parameters
        ----------
        obj : Any
            Value for ``obj``.

        Returns
        -------
        Any
            Result of the operation.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class CalibrationStore:
    """
    Persistent key-value store for calibration parameters.

    Each entry carries a timestamp so callers can detect stale calibrations
    via ``is_stale()``.  Data is persisted as JSON on disk so values survive
    process restarts.

    Parameters
    ----------
    path : str
        Path to the JSON file used for persistent storage.
        A new file is created if it does not exist.
    default_max_age_hours : float
        Default staleness threshold in hours (default 24).
    """

    def __init__(self, path: str, default_max_age_hours: float = 24.0):
        """Initialize the CalibrationStore instance.

        Parameters
        ----------
        path : str
            Filesystem path.
        default_max_age_hours : float, default: 24.0
            Value for ``default_max_age_hours``.
        """
        self._path = path
        self._default_max_age = timedelta(hours=default_max_age_hours)
        self._store: dict[str, dict[str, Any]] = {}
        self._load()


    def _load(self):
        """Return the load result."""
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def save(self):
        """Flush the in-memory store to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._store, f, indent=2, cls=_NumpyEncoder)


    def set(self, qubit: str, key: str, value: Any, *, autosave: bool = True):
        """Store a calibration parameter for *qubit*.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        key : str
            Lookup key.
        value : Any
            Value to apply.
        autosave : bool, default: True
            Value for ``autosave``.
        """
        if qubit not in self._store:
            self._store[qubit] = {}
        self._store[qubit][key] = {
            "value": value if not isinstance(value, np.ndarray) else value.tolist(),
            "timestamp": datetime.now().isoformat(),
        }
        if autosave:
            self.save()

    def get(self, qubit: str, key: str, default: Any = None) -> Any:
        """Retrieve the stored value (without timestamp metadata).

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        key : str
            Lookup key.
        default : Any, default: None
            Value for ``default``.

        Returns
        -------
        Any
            Result of the operation.
        """
        try:
            return self._store[qubit][key]["value"]
        except KeyError:
            return default

    def get_with_meta(self, qubit: str, key: str) -> dict | None:
        """Return the full record ``{value, timestamp}`` or None.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        key : str
            Lookup key.

        Returns
        -------
        dict | None
            Result of the operation.
        """
        try:
            return self._store[qubit][key]
        except KeyError:
            return None

    def timestamp(self, qubit: str, key: str) -> datetime | None:
        """Return the datetime when *key* was last set, or None.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        key : str
            Lookup key.

        Returns
        -------
        datetime | None
            Result of the operation.
        """
        rec = self.get_with_meta(qubit, key)
        if rec is None:
            return None
        try:
            return datetime.fromisoformat(rec["timestamp"])
        except (ValueError, KeyError):
            return None

    def is_stale(self, qubit: str, key: str, max_age_hours: float | None = None) -> bool:
        """Return True if *key* is missing or older than *max_age_hours*.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        key : str
            Calibration parameter key.
        max_age_hours : float or None
            Override the instance-level default staleness threshold.

        Returns
        -------
        bool
            Result of the operation.
        """
        ts = self.timestamp(qubit, key)
        if ts is None:
            return True
        threshold = timedelta(hours=max_age_hours) if max_age_hours is not None else self._default_max_age
        return datetime.now() - ts > threshold

    def all_keys(self, qubit: str) -> list[str]:
        """Return all parameter keys stored for *qubit*.

        Parameters
        ----------
        qubit : str
            Qubit identifier.

        Returns
        -------
        list[str]
            Result of the operation.
        """
        return list(self._store.get(qubit, {}).keys())

    def all_qubits(self) -> list[str]:
        """Return the all qubits result.

        Returns
        -------
        list[str]
            Result of the operation.
        """
        return list(self._store.keys())

    def delete(self, qubit: str, key: str, *, autosave: bool = True):
        """Remove a single entry.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        key : str
            Lookup key.
        autosave : bool, default: True
            Value for ``autosave``.
        """
        if qubit in self._store and key in self._store[qubit]:
            del self._store[qubit][key]
            if autosave:
                self.save()

    def clear_qubit(self, qubit: str, *, autosave: bool = True):
        """Remove all entries for *qubit*.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        autosave : bool, default: True
            Value for ``autosave``.
        """
        self._store.pop(qubit, None)
        if autosave:
            self.save()


    def update_from_dict(self, qubit: str, params: dict[str, Any], *, autosave: bool = True):
        """Batch-update multiple parameters for *qubit* from a flat dict.

        Parameters
        ----------
        qubit : str
            Qubit identifier.
        params : dict[str, Any]
            Value for ``params``.
        autosave : bool, default: True
            Value for ``autosave``.
        """
        for key, value in params.items():
            self.set(qubit, key, value, autosave=False)
        if autosave:
            self.save()

    def to_flat_dict(self, qubit: str) -> dict[str, Any]:
        """Return all values for *qubit* as a plain ``{key: value}`` dict.

        Parameters
        ----------
        qubit : str
            Qubit identifier.

        Returns
        -------
        dict[str, Any]
            Result of the operation.
        """
        return {k: v["value"] for k, v in self._store.get(qubit, {}).items()}

    def summary(self, qubit: str | None = None) -> str:
        """Return a summary of the current state.

        Parameters
        ----------
        qubit : str | None, default: None
            Qubit identifier.

        Returns
        -------
        str
            Result of the operation.
        """
        qubits = [qubit] if qubit is not None else self.all_qubits()
        lines = []
        for q in qubits:
            lines.append(f"[{q}]")
            for k, rec in self._store.get(q, {}).items():
                stale = " (STALE)" if self.is_stale(q, k) else ""
                lines.append(f"  {k:<28s} = {rec['value']}  @ {rec['timestamp']}{stale}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a human-readable representation.

        Returns
        -------
        str
            Result of the operation.
        """
        total = sum(len(v) for v in self._store.values())
        return f"CalibrationStore(path={self._path!r}, qubits={self.all_qubits()}, entries={total})"
