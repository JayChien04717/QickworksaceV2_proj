"""Named, lossless IQ data and independently versioned analysis results."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4
import numpy as np

from labtools.fitting.result import FitResult
from labtools.hdf5 import save_hdf5, read_hdf5, result_record


class QualityFlag(str, Enum):
    GOOD = "good"
    BAD = "bad"
    NO_INFORMATION = "not_analyzed"


@dataclass
class TraceData:
    iq: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    units: dict[str, str] = field(default_factory=dict)
    shots: np.ndarray | None = None
    shot_dims: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.iq = np.asarray(self.iq, dtype=np.complex128)
        self.dims = tuple(self.dims)
        self.coords = {k: np.asarray(v) for k, v in self.coords.items()}
        if len(set(self.dims)) != len(self.dims) or len(self.dims) != self.iq.ndim:
            raise ValueError("IQ dimensions must be unique and match array rank")
        for name, size in zip(self.dims, self.iq.shape):
            if name not in self.coords or self.coords[name].ndim != 1 or len(self.coords[name]) != size:
                raise ValueError(f"Coordinate {name} does not match IQ shape")
        if self.shots is not None:
            self.shots = np.asarray(self.shots, dtype=np.complex128)
            self.shot_dims = tuple(self.shot_dims)
            if len(self.shot_dims) != self.shots.ndim or len(set(self.shot_dims)) != len(self.shot_dims):
                raise ValueError("Shot dimensions do not match shot array")
            for dim, size in zip(self.shot_dims, self.shots.shape):
                if dim != "shot" and (dim not in self.coords or len(self.coords[dim]) != size):
                    raise ValueError(f"Shot coordinate {dim} does not match shape")

    def signal(self, mode="abs", *, rotation_deg=0):
        if mode == "population":
            threshold = self.metadata.get("threshold")
            if threshold is None:
                raise ValueError("Population requires an explicit ro_threshold and captured shots")
            return self.population(threshold, rotation_deg=rotation_deg)
        iq = self.iq * np.exp(-1j * np.deg2rad(rotation_deg))
        def decibels(value):
            with np.errstate(divide="ignore", invalid="ignore"):
                return 20*np.log10(np.abs(value))
        funcs = {"abs": np.abs, "real": np.real, "imag": np.imag, "phase": np.angle, "db": decibels}
        if mode not in funcs:
            raise ValueError(f"Unknown IQ processing {mode}")
        return funcs[mode](iq)

    def population(self, threshold, *, rotation_deg=0):
        """Classify each retained shot before averaging; never threshold averaged IQ."""
        if self.shots is None or "shot" not in self.shot_dims:
            raise ValueError("Population requires captured shots")
        if not np.isfinite(threshold) or not np.isfinite(rotation_deg):
            raise ValueError("Threshold and rotation must be finite")
        if not np.isfinite(self.shots).all():
            raise ValueError("Population requires finite shots")
        projected = (self.shots * np.exp(-1j * np.deg2rad(rotation_deg))).real
        return np.mean(projected > threshold, axis=self.shot_dims.index("shot"))

    def select(self, **indices):
        """Select integer indices without guessing what positional dimensions mean."""
        slices = tuple(indices.get(d, slice(None)) for d in self.dims)
        unknown = set(indices) - set(self.dims)
        if unknown:
            raise KeyError(unknown)
        return self.iq[slices]


@dataclass
class ExperimentData:
    experiment: str
    traces: dict[str, TraceData]
    metadata: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fits: dict[str, FitResult] = field(default_factory=dict)
    analysis_status: str = "not_analyzed"
    analysis_message: str = ""
    path: Path | None = None

    def __getitem__(self, target):
        return self.traces[target]

    @property
    def targets(self):
        return tuple(self.traces)

    @property
    def metrics(self):
        return {q: fit.parameters.copy() for q, fit in self.fits.items()}

    def is_good(self):
        return self.analysis_status == "completed" and bool(self.fits) and all(f.success for f in self.fits.values())

    @property
    def quality(self):
        if self.analysis_status == "not_analyzed":
            return QualityFlag.NO_INFORMATION
        return QualityFlag.GOOD if self.is_good() else QualityFlag.BAD

    def plot(self, **kwargs):
        if getattr(self, "_plotter", None):
            return self._plotter(self, **kwargs)
        from QickworkspaceV2.plotting import plot_result
        return plot_result(self, **kwargs)

    def save_labber(self, path=None, **options):
        """Export a Labber log with complete V2 data embedded in /metagroup."""
        from labtools.labber import save_labber
        return save_labber(self, path, **options)

    def save(self, path, *, catalog_root=None):
        """Save the native record; file I/O belongs to labtools."""
        self.path = save_hdf5(path, result_record(self), self.traces, catalog_root=catalog_root)
        return self.path

    @classmethod
    def load(cls, path):
        record, traces = read_hdf5(path)
        record["fits"] = {q: FitResult(**v) for q, v in record["fits"].items()}
        return cls(traces={q: TraceData(**v) for q, v in traces.items()},
                   path=Path(path).resolve(), **record)
