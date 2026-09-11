"""Named, lossless IQ data and independently versioned analysis results."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4
import json
import os
import tempfile
import numpy as np

from QickworkspaceV2.data.serialization import dumps
from QickworkspaceV2.data.atomic import replace_file


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
class FitResult:
    model: str
    success: bool
    parameters: dict[str, float] = field(default_factory=dict)
    errors: dict[str, float | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    r_squared: float | None = None
    message: str = ""
    x_fit: list[float] = field(default_factory=list)
    y_fit: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)


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
        from QickworkspaceV2.data.labber import save_labber
        return save_labber(self, path, **options)

    def save(self, path):
        """Atomic standalone HDF5, preserving complex data and every named axis."""
        import h5py
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".run-", suffix=".h5", dir=path.parent)
        os.close(fd)
        try:
            with h5py.File(temp, "w") as f:
                f.attrs["schema_version"] = 2
                f.attrs["record"] = dumps({"experiment": self.experiment, "metadata": self.metadata,
                                            "run_id": self.run_id, "created_at": self.created_at,
                                            "fits": self.fits, "analysis_status": self.analysis_status,
                                            "analysis_message": self.analysis_message})
                trace_group = f.create_group("traces", track_order=True)
                for name, trace in self.traces.items():
                    group = trace_group.create_group(name)
                    group.attrs["record"] = dumps({"dims": trace.dims, "units": trace.units,
                                                   "shot_dims": trace.shot_dims, "metadata": trace.metadata})
                    group.create_dataset("iq", data=trace.iq, compression="gzip")
                    if trace.shots is not None:
                        group.create_dataset("shots", data=trace.shots, compression="gzip")
                    for dim, coord in trace.coords.items():
                        if coord.dtype.kind in "UO":
                            group.create_dataset(f"coords/{dim}", data=coord.astype(object), dtype=h5py.string_dtype())
                        else:
                            group.create_dataset(f"coords/{dim}", data=coord)
                f.flush()
            replace_file(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.path = path.resolve()
        return self.path

    @classmethod
    def load(cls, path):
        import h5py
        with h5py.File(path, "r") as f:
            if f.attrs.get("schema_version") != 2:
                raise ValueError("Unsupported HDF5 schema; use the legacy data reader for v1 files")
            record = json.loads(f.attrs["record"])
            record["fits"] = {q: FitResult(**v) for q, v in record["fits"].items()}
            traces = {}
            for name, group in f["traces"].items():
                meta = json.loads(group.attrs["record"])
                coords = {d: v.asstr()[...] if v.dtype.kind in "OS" else v[...] for d, v in group["coords"].items()}
                traces[name] = TraceData(group["iq"][...], coords=coords,
                                          shots=group["shots"][...] if "shots" in group else None, **meta)
        return cls(traces=traces, path=Path(path).resolve(), **record)
