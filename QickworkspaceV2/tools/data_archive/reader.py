"""Selective, dimension-aware reader for native Qickworkspace HDF5 files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from ..hdf5_store import SCHEMA_NAME, inspect_file, load_result


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"O", "S"}:
        return np.asarray([
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in value.reshape(-1)
        ]).reshape(value.shape)
    return value


def _attribute(value):
    value = _decode(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dims(dataset: h5py.Dataset) -> tuple[str, ...]:
    value = dataset.attrs.get("dims", "[]")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    return tuple(str(item) for item in (value or []))


@dataclass(frozen=True)
class LabeledArray:
    """A NumPy array accompanied by its HDF5 dimension and axis metadata."""

    values: Any
    dims: tuple[str, ...] = ()
    axes: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    dataset: str = ""

    @property
    def shape(self):
        return np.shape(self.values)

    @property
    def dtype(self):
        return np.asarray(self.values).dtype

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


class ExperimentReader:
    """Read one experiment without loading unrelated large datasets."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def inspect(self) -> dict:
        return inspect_file(self.path)

    def load(self):
        """Load the complete experiment into an ExperimentData object."""
        return load_result(self.path)

    def axes(self) -> dict[str, dict[str, Any]]:
        """Load the small axis arrays and their labels/units."""
        result = {}
        with self._open() as h5:
            for name, group in h5["axes"].items():
                result[name] = {
                    "values": _decode(group["values"][()]),
                    **{key: _attribute(value) for key, value in group.attrs.items()},
                }
        return result

    def raw_keys(self) -> list[str]:
        return self._dataset_keys("raw")

    def analysis_keys(self) -> list[str]:
        return self._dataset_keys("analysis")

    def raw(
        self,
        name: str = "iq",
        *,
        selection: Optional[Mapping[str, Any] | tuple] = None,
    ) -> LabeledArray:
        return self._read_dataset("raw", name, selection)

    def analysis(
        self,
        name: str,
        *,
        selection: Optional[Mapping[str, Any] | tuple] = None,
    ) -> LabeledArray:
        return self._read_dataset("analysis", name, selection)

    def _open(self):
        h5 = h5py.File(self.path, "r")
        if h5.attrs.get("schema_name") != SCHEMA_NAME:
            h5.close()
            raise ValueError(f"Selective reading requires native HDF5 v1: {self.path}")
        if not bool(h5.attrs.get("write_complete", False)):
            h5.close()
            raise ValueError(f"Experiment file is incomplete: {self.path}")
        return h5

    def _dataset_keys(self, group_name: str) -> list[str]:
        keys = []
        with self._open() as h5:
            h5[group_name].visititems(
                lambda name, node: keys.append(name) if isinstance(node, h5py.Dataset) else None
            )
        return sorted(keys)

    @staticmethod
    def _clean_name(name: str, group_name: str) -> str:
        name = str(name).strip("/")
        prefix = f"{group_name}/"
        if name.startswith(prefix):
            name = name[len(prefix):]
        if not name or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError(f"Invalid dataset name: {name!r}")
        return name

    def _read_dataset(self, group_name: str, name: str, selection) -> LabeledArray:
        name = self._clean_name(name, group_name)
        with self._open() as h5:
            path = f"{group_name}/{name}"
            if path not in h5 or not isinstance(h5[path], h5py.Dataset):
                available = self._keys_in_open_file(h5[group_name])
                raise KeyError(f"Dataset {path!r} not found. Available: {available}")
            dataset = h5[path]
            dims = _dims(dataset)
            index = self._selection_index(h5, dataset, dims, selection)
            values = _decode(dataset[index])
            remaining_dims = tuple(
                dim for dim, item in zip(dims, index)
                if not isinstance(item, (int, np.integer))
            )
            axes = {}
            for dim, item in zip(dims, index):
                if isinstance(item, (int, np.integer)) or f"axes/{dim}/values" not in h5:
                    continue
                axes[dim] = _decode(h5[f"axes/{dim}/values"][item])
            attrs = {key: _attribute(value) for key, value in dataset.attrs.items()}
            return LabeledArray(values, remaining_dims, axes, attrs, path)

    @staticmethod
    def _keys_in_open_file(group: h5py.Group) -> list[str]:
        keys = []
        group.visititems(
            lambda name, node: keys.append(name) if isinstance(node, h5py.Dataset) else None
        )
        return sorted(keys)

    def _selection_index(self, h5, dataset, dims, selection):
        if selection is None:
            return tuple(slice(None) for _ in range(dataset.ndim))
        if isinstance(selection, tuple):
            if len(selection) > dataset.ndim:
                raise IndexError("Selection has more indices than the dataset rank")
            return selection + tuple(slice(None) for _ in range(dataset.ndim - len(selection)))
        if not isinstance(selection, Mapping):
            raise TypeError("selection must be a dims mapping or an index tuple")
        unknown = set(selection) - set(dims)
        if unknown:
            raise KeyError(f"Unknown dimensions {sorted(unknown)}; available: {list(dims)}")
        index = []
        for dim in dims:
            item = selection.get(dim, slice(None))
            if isinstance(item, str):
                axis_path = f"axes/{dim}/values"
                if axis_path not in h5:
                    raise KeyError(f"Dimension {dim!r} has no axis labels")
                labels = _decode(h5[axis_path][()]).reshape(-1)
                matches = np.flatnonzero(labels.astype(str) == item)
                if not len(matches):
                    raise KeyError(f"{item!r} is not present on axis {dim!r}")
                item = int(matches[0])
            index.append(item)
        return tuple(index)
