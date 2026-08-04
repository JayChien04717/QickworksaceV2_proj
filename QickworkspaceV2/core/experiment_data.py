"""
ExperimentData — unified result container for all experiments.

Backward compatibility
----------------------
Old code that unpacks ``(fit_params, error) = expt.run()`` still works via
``__iter__``.  Old code that does ``freq = expt.run()`` still works via
``__float__`` / ``__int__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np


def _new_experiment_id() -> str:
    """Return the new experiment id result.

    Returns
    -------
    str
        Result of the operation.
    """
    from ..tools.hdf5_store import generate_experiment_id

    return generate_experiment_id()


class QualityFlag(Enum):
    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    NO_INFORMATION = "no_information"


@dataclass
class ExperimentData:
    """
    Unified result container returned by every ``BaseExperiment.run()`` call.

    Backward Compatibility
    ----------------------
    * ``fit_params, error = result``  — tuple unpacking via ``__iter__``
    * ``float(result)``               — scalar result via ``__float__``
    * ``result[0]``, ``result[1]``    — index access via ``__getitem__``

    New API
    -------
    * ``result.fit_result``     — named dict of fitted parameters + uncertainties
    * ``result.quality``        — ``QualityFlag`` enum
    * ``result.is_good()``      — convenience boolean
    * ``result.to_dict()``      — JSON-serialisable dict
    * ``result.save(path)``     — HDF5 save
    * ``ExperimentData.load(path)`` — HDF5 load
    """

    experiment_type: str = ""
    experiment_id: str = field(default_factory=_new_experiment_id)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    raw_iq: Any = None
    x_axis: Optional[np.ndarray] = None
    y_axis: Optional[np.ndarray] = None

    # Dimension-aware native HDF5 payload. Existing raw_iq/x_axis/y_axis remain
    # the compatibility surface for ordinary 1D/2D experiments.
    axes: dict = field(default_factory=dict)
    raw_data: dict = field(default_factory=dict)
    analysis_data: dict = field(default_factory=dict)
    dataset_dims: dict = field(default_factory=dict)

    fit_params: Optional[np.ndarray] = None
    fit_errors: Optional[np.ndarray] = None

    # Named results dict — new API; keyed by param name, value is (val, err)
    fit_result: dict = field(default_factory=dict)

    # Scalar result for experiments that return a single number (e.g. frequency)
    scalar_result: Optional[float] = None

    figures: list = field(default_factory=list)

    quality: QualityFlag = QualityFlag.NO_INFORMATION
    quality_message: str = ""

    # Legacy config metadata. New experiment runs leave this empty; config
    # management and presentation remain owned by ExperimentConfig.
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # Native HDF5 discovery and presentation metadata
    data_kind: str = ""
    analysis_id: str = ""
    plot_id: str = ""
    comment: str = ""
    tags: list = field(default_factory=list)
    session_id: Optional[str] = None

    parent_id: Optional[str] = None
    children: list = field(default_factory=list)

    x_name: str = ""
    x_unit: str = ""
    x_scale: float = 1.0
    y_name: str = ""
    y_unit: str = ""
    y_scale: float = 1.0

    interrupted: bool = False
    avg_count: int = 0


    def __iter__(self):
        """Support ``fit_params, error = result``.

        Yields
        ------
        Any
            Values produced by the iterator.
        """
        yield self.fit_params
        yield self.fit_errors

    def __getitem__(self, idx):
        """String key  → ``result['pi_gain']``  shortcut for ``fit_result['pi_gain'][0]``.
                        Integer idx → ``result[0]`` (fit_params), ``result[1]`` (fit_errors).

        Parameters
        ----------
        idx : Any
            Value for ``idx``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        KeyError
            If the operation cannot be completed.
        """
        if isinstance(idx, str):
            entry = self.fit_result.get(idx)
            if entry is None:
                raise KeyError(
                    f"'{idx}' not in fit_result. Available: {list(self.fit_result)}"
                )
            return entry[0] if isinstance(entry, (tuple, list)) else entry
        return (self.fit_params, self.fit_errors)[idx]

    def __float__(self):
        """Support ``freq = float(result)`` for single-value experiments.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        TypeError
            If the operation cannot be completed.
        """
        if self.scalar_result is not None:
            return float(self.scalar_result)
        if self.fit_params is not None and len(self.fit_params) > 0:
            return float(self.fit_params[0])
        raise TypeError(f"ExperimentData '{self.experiment_type}' has no scalar result")

    def __bool__(self):
        """True when data was acquired and fit succeeded.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self.raw_iq is not None and self.fit_params is not None


    def is_good(self) -> bool:
        """Return whether is good.

        Returns
        -------
        bool
            Result of the operation.
        """
        return self.quality == QualityFlag.GOOD

    def get_param(self, name: str, default=None):
        """Return named fit result value, or default.

        Parameters
        ----------
        name : str
            Name of the target object.
        default : Any, default: None
            Value for ``default``.

        Returns
        -------
        Any
            Result of the operation.
        """
        entry = self.fit_result.get(name)
        if entry is None:
            return default
        return entry[0] if isinstance(entry, (tuple, list)) else entry

    def get_error(self, name: str, default=None):
        """Return named fit result uncertainty, or default.

        Parameters
        ----------
        name : str
            Name of the target object.
        default : Any, default: None
            Value for ``default``.

        Returns
        -------
        Any
            Result of the operation.
        """
        entry = self.fit_result.get(name)
        if isinstance(entry, (tuple, list)) and len(entry) > 1:
            return entry[1]
        return default


    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (numpy arrays → lists).

        Returns
        -------
        dict
            Result of the operation.
        """

        def _arr(v):
            """Return the arr result.

            Parameters
            ----------
            v : Any
                Value for ``v``.

            Returns
            -------
            Any
                Result of the operation.
            """
            return v.tolist() if isinstance(v, np.ndarray) else v

        return {
            "experiment_type": self.experiment_type,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp.isoformat(),
            "fit_params": _arr(self.fit_params),
            "fit_errors": _arr(self.fit_errors),
            "fit_result": {
                k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v)
                for k, v in self.fit_result.items()
            },
            "scalar_result": self.scalar_result,
            "quality": self.quality.value,
            "quality_message": self.quality_message,
            "config": self.config,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "x_name": self.x_name,
            "x_unit": self.x_unit,
            "x_scale": self.x_scale,
            "y_name": self.y_name,
            "y_unit": self.y_unit,
            "y_scale": self.y_scale,
            "interrupted": self.interrupted,
            "avg_count": self.avg_count,
            "data_kind": self.data_kind,
            "analysis_id": self.analysis_id,
            "plot_id": self.plot_id,
            "comment": self.comment,
            "tags": list(self.tags),
            "session_id": self.session_id,
            "dataset_dims": self.dataset_dims,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentData":
        """Return the from dict result.

        Parameters
        ----------
        d : dict
            Value for ``d``.

        Returns
        -------
        'ExperimentData'
            Result of the operation.
        """
        obj = cls(
            experiment_type=d.get("experiment_type", ""),
            experiment_id=d.get("experiment_id") or _new_experiment_id(),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(timezone.utc),
            fit_params=np.array(d["fit_params"]) if d.get("fit_params") is not None else None,
            fit_errors=np.array(d["fit_errors"]) if d.get("fit_errors") is not None else None,
            fit_result=d.get("fit_result", {}),
            scalar_result=d.get("scalar_result"),
            quality=QualityFlag(d.get("quality", "no_information")),
            quality_message=d.get("quality_message", ""),
            config=d.get("config", {}),
            metadata=d.get("metadata", {}),
            parent_id=d.get("parent_id"),
            x_name=d.get("x_name", ""),
            x_unit=d.get("x_unit", ""),
            x_scale=d.get("x_scale", 1.0),
            y_name=d.get("y_name", ""),
            y_unit=d.get("y_unit", ""),
            y_scale=d.get("y_scale", 1.0),
            interrupted=d.get("interrupted", False),
            avg_count=d.get("avg_count", 0),
            data_kind=d.get("data_kind", ""),
            analysis_id=d.get("analysis_id", ""),
            plot_id=d.get("plot_id", ""),
            comment=d.get("comment", ""),
            tags=d.get("tags", []),
            session_id=d.get("session_id"),
            dataset_dims=d.get("dataset_dims", {}),
        )
        return obj

    def save(
        self,
        filepath: Optional[str] = None,
        *,
        comment: str = "",
        tags=(),
        data_root: Optional[str] = None,
        catalog: bool = True,
    ):
        """Save through the native HDF5 v1 writer and update its catalog.

        Parameters
        ----------
        filepath : Optional[str]
            Value for ``filepath``.
        comment : str, default: ''
            Value for ``comment``.
        tags : Any, default: ()
            Value for ``tags``.
        data_root : Optional[str]
            Value for ``data_root``.
        catalog : bool, default: True
            Value for ``catalog``.

        Returns
        -------
        Any
            Result of the operation.
        """
        from ..tools.hdf5_store import save_result

        return save_result(
            self,
            filepath,
            comment=comment,
            tags=tags,
            data_root=data_root,
            catalog=catalog,
        )

    @classmethod
    def load(cls, filepath: str) -> "ExperimentData":
        """Load native v1 or the previous local ExperimentData HDF5 format.

        Parameters
        ----------
        filepath : str
            Value for ``filepath``.

        Returns
        -------
        'ExperimentData'
            Result of the operation.
        """
        from ..tools.hdf5_store import load_result

        return load_result(filepath)

    def __repr__(self) -> str:
        """Return a human-readable representation.

        Returns
        -------
        str
            Result of the operation.
        """
        status = "interrupted" if self.interrupted else "complete"
        return (
            f"ExperimentData(type={self.experiment_type!r}, "
            f"id={self.experiment_id!r}, quality={self.quality.value}, "
            f"status={status})"
        )
