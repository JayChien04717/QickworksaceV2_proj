"""Native Qickworkspace HDF5 persistence and searchable experiment catalog.

The HDF5 file is the source of truth.  The adjacent SQLite catalog contains
only lightweight, rebuildable metadata used to locate files without opening
every dataset.
"""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import h5py
import numpy as np


SCHEMA_NAME = "qickworkspace.experiment"
SCHEMA_VERSION = "1.0"
LOCAL_TIMEZONE = ZoneInfo("Asia/Taipei")
CATALOG_FILENAME = "catalog.sqlite"
_ID_RE = re.compile(r"^\d{8}T\d{12}Z-[0-9A-HJKMNP-TV-Z]{13}$")
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def _encode_crockford(value: int, width: int) -> str:
    chars = []
    for _ in range(width):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def _as_utc(value: Optional[datetime] = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOCAL_TIMEZONE)
    return value.astimezone(timezone.utc)


def generate_experiment_id(timestamp: Optional[datetime] = None) -> str:
    """Return a sortable UTC timestamp plus 64 bits of cryptographic randomness."""
    stamp = _as_utc(timestamp).strftime("%Y%m%dT%H%M%S%fZ")
    random_part = _encode_crockford(int.from_bytes(secrets.token_bytes(8), "big"), 13)
    return f"{stamp}-{random_part}"


def validate_experiment_id(experiment_id: str) -> bool:
    """Return whether *experiment_id* follows the v1 public ID format."""
    return bool(_ID_RE.fullmatch(str(experiment_id or "")))


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return repr(value)


def _json_object_hook(value: dict):
    marker = value.get("__complex__")
    if isinstance(marker, list) and len(marker) == 2:
        return complex(marker[0], marker[1])
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def _json_loads(value: Any, default=None):
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(str(value), object_hook=_json_object_hook)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _write_text(group: h5py.Group, name: str, value: Any) -> h5py.Dataset:
    return group.create_dataset(name, data=str(value or ""), dtype=_STRING_DTYPE)


def _read_text(group: h5py.Group, name: str, default: str = "") -> str:
    if name not in group:
        return default
    value = group[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _dataset_options(array: np.ndarray) -> dict:
    if array.ndim == 0 or array.size < 64 or array.dtype.kind in {"O", "U", "S"}:
        return {}
    return {"chunks": True, "compression": "gzip", "compression_opts": 4, "shuffle": True}


def _normalise_dataset_payload(value: Any):
    if isinstance(value, dict) and "values" in value:
        attrs = {key: value.get(key) for key in ("dims", "unit", "label", "description", "scale")}
        return value["values"], attrs
    return value, {}


def _write_tree(group: h5py.Group, values: Any, dataset_name: str = "value") -> None:
    if values is None:
        return
    is_payload = isinstance(values, dict) and "values" in values
    if isinstance(values, dict) and not is_payload:
        for key, value in values.items():
            safe_key = str(key).replace("/", "_")
            if isinstance(value, dict) and "values" not in value:
                _write_tree(group.require_group(safe_key), value)
            else:
                _write_tree(group, value, safe_key)
        return

    values, attrs = _normalise_dataset_payload(values)
    if values is None:
        return
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        if array.ndim == 0:
            dataset = group.create_dataset(dataset_name, data=str(array.item()), dtype=_STRING_DTYPE)
        elif all(isinstance(item, (str, bytes, np.str_)) for item in array.reshape(-1)):
            flat = [item.decode() if isinstance(item, bytes) else str(item) for item in array.reshape(-1)]
            dataset = group.create_dataset(dataset_name, data=np.asarray(flat, dtype=_STRING_DTYPE).reshape(array.shape))
        else:
            dataset = group.create_dataset(dataset_name, data=_json_dumps(values), dtype=_STRING_DTYPE)
            dataset.attrs["encoding"] = "json"
    else:
        dataset = group.create_dataset(dataset_name, data=array, **_dataset_options(array))
    for key, value in attrs.items():
        if value not in (None, "", []):
            dataset.attrs[key] = _json_dumps(value) if isinstance(value, (list, tuple, dict)) else value


def _read_tree(node: h5py.Group | h5py.Dataset, *, preserve_attrs: bool = False):
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if node.attrs.get("encoding") == "json":
            value = _json_loads(value)
        elif isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, np.ndarray) and value.dtype.kind in {"O", "S"}:
            value = np.asarray([
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in value.reshape(-1)
            ]).reshape(value.shape)
        if preserve_attrs:
            attrs = {}
            for key in ("dims", "unit", "label", "description"):
                if key in node.attrs:
                    raw = node.attrs[key]
                    attrs[key] = _json_loads(raw, raw) if key == "dims" else raw
            if attrs:
                return {"values": value, **attrs}
        return value
    return {key: _read_tree(child, preserve_attrs=preserve_attrs) for key, child in node.items()}


def _collect_dataset_dims(group: h5py.Group, prefix: str = "") -> dict[str, list[str]]:
    dims = {}
    for name, node in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(node, h5py.Group):
            dims.update(_collect_dataset_dims(node, path))
        elif "dims" in node.attrs:
            parsed = _json_loads(node.attrs["dims"], [])
            dims[path] = list(parsed or [])
    return dims


def _decorate_tree_with_dims(values: Any, dims: dict[str, list[str]], prefix: str = ""):
    if isinstance(values, dict) and "values" in values:
        if "dims" not in values and prefix in dims:
            return {**values, "dims": dims[prefix]}
        return values
    if isinstance(values, dict):
        return {
            key: _decorate_tree_with_dims(
                value,
                dims,
                f"{prefix}/{key}" if prefix else str(key),
            )
            for key, value in values.items()
        }
    if prefix in dims:
        return {"values": values, "dims": dims[prefix]}
    return values


def _quality_value(result) -> str:
    quality = getattr(result, "quality", "no_information")
    return str(getattr(quality, "value", quality))


def _normalise_tags(tags: Iterable[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]
    return list(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))


def _qubits_from_result(result) -> list[str]:
    metadata = getattr(result, "metadata", {}) or {}
    config = getattr(result, "config", {}) or {}
    raw = metadata.get("qubit_names") or metadata.get("qubits")
    if raw is None:
        raw = metadata.get("qubit") or config.get("name") or config.get("qubit")
    if raw is None:
        return []
    if isinstance(raw, (str, int)):
        raw = [raw]
    return [str(item) for item in raw]


def _dispatch_ids(result) -> tuple[str, str, str]:
    experiment_type = str(getattr(result, "experiment_type", "") or "").lower()
    raw = getattr(result, "raw_iq", None)
    ndim = np.asarray(raw).ndim if raw is not None and not isinstance(raw, dict) else 0
    rules = (
        ("single", "single_shot", "single_shot", "single_shot_iq"),
        ("tomography", "tomography", "tomography", "density_matrix"),
        ("rb", "rb", "rb", "rb_decay"),
        ("rabi", "sweep_1d", "rabi", "iq_fit_1d"),
        ("t1", "sweep_1d", "t1", "iq_fit_1d"),
        ("ramsey", "sweep_1d", "ramsey", "iq_fit_1d"),
        ("echo", "sweep_1d", "spin_echo", "iq_fit_1d"),
        ("spec", "sweep_1d" if ndim <= 1 else "sweep_nd", "spectroscopy", "iq_fit_1d"),
    )
    inferred = ("sweep_nd" if ndim > 1 else "sweep_1d", "generic", "generic_iq")
    for token, data_kind, analysis_id, plot_id in rules:
        if token in experiment_type:
            inferred = (data_kind, analysis_id, plot_id)
            break
    return (
        str(getattr(result, "data_kind", "") or inferred[0]),
        str(getattr(result, "analysis_id", "") or inferred[1]),
        str(getattr(result, "plot_id", "") or inferred[2]),
    )


def _resolve_data_root(data_root: Optional[os.PathLike | str]) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    try:
        from ..core.base_experiment import BaseExperiment

        configured = BaseExperiment._data_path
    except Exception:
        configured = None
    return Path(configured or "data").expanduser().resolve()


def _safe_filename_part(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-_.")
    return cleaned or fallback


def _auto_path(root: Path, result, local_time: datetime, experiment_id: str) -> Path:
    experiment = _safe_filename_part(getattr(result, "experiment_type", ""), "experiment")
    qubits = _qubits_from_result(result)
    qubit = _safe_filename_part("-".join(qubits), "all")
    folder = root / local_time.strftime("%Y") / local_time.strftime("%m") / local_time.strftime("%Y-%m-%d")
    short_id = experiment_id.rsplit("-", 1)[-1][-8:]
    filename = f"{local_time:%H%M%S_%f}_{experiment}_{qubit}_{short_id}.h5"
    return folder / filename


def _axis_entries(result) -> dict[str, Any]:
    entries = dict(getattr(result, "axes", {}) or {})
    if getattr(result, "x_axis", None) is not None and "x" not in entries:
        entries["x"] = {
            "values": np.asarray(result.x_axis) * float(getattr(result, "x_scale", 1.0)),
            "label": getattr(result, "x_name", ""),
            "unit": getattr(result, "x_unit", ""),
            "scale": float(getattr(result, "x_scale", 1.0)),
        }
    if getattr(result, "y_axis", None) is not None and "y" not in entries:
        entries["y"] = {
            "values": np.asarray(result.y_axis) * float(getattr(result, "y_scale", 1.0)),
            "label": getattr(result, "y_name", ""),
            "unit": getattr(result, "y_unit", ""),
            "scale": float(getattr(result, "y_scale", 1.0)),
        }
    return entries


def _provenance() -> dict:
    try:
        from .. import __version__
    except Exception:
        __version__ = "unknown"
    try:
        import qick
        qick_version = getattr(qick, "__version__", "unknown")
    except Exception:
        qick_version = None
    return {
        "qickworkspace_version": __version__,
        "qick_version": qick_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _write_file(path: Path, result, *, comment: str, tags: list[str], utc_time: datetime) -> None:
    local_time = utc_time.astimezone(LOCAL_TIMEZONE)
    data_kind, analysis_id, plot_id = _dispatch_ids(result)
    metadata = dict(getattr(result, "metadata", {}) or {})
    session_id = str(getattr(result, "session_id", "") or metadata.get("session_id", ""))
    lineage = {
        "session_id": session_id or None,
        "parent_id": getattr(result, "parent_id", None),
        "children": getattr(result, "children", []) or [],
    }
    with h5py.File(path, "w") as h5:
        h5.attrs.update({
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "write_complete": False,
            "experiment_id": result.experiment_id,
            "experiment_type": str(getattr(result, "experiment_type", "")),
            "timestamp_utc": utc_time.isoformat(),
            "timestamp_local": local_time.isoformat(),
            "data_kind": data_kind,
            "analysis_id": analysis_id,
            "plot_id": plot_id,
            "quality": _quality_value(result),
            "interrupted": bool(getattr(result, "interrupted", False)),
        })

        meta = h5.create_group("meta")
        _write_text(meta, "comment", comment)
        meta.create_dataset("tags", data=np.asarray(tags, dtype=_STRING_DTYPE))
        _write_text(meta, "config_json", _json_dumps(getattr(result, "config", {}) or {}))
        _write_text(meta, "metadata_json", _json_dumps(metadata))
        _write_text(meta, "provenance_json", _json_dumps(_provenance()))
        _write_text(meta, "lineage_json", _json_dumps(lineage))

        axes = h5.create_group("axes")
        for name, payload in _axis_entries(result).items():
            values, attrs = _normalise_dataset_payload(payload)
            axis_group = axes.create_group(str(name).replace("/", "_"))
            array = np.asarray(values)
            if array.dtype.kind in {"O", "U", "S"}:
                strings = [item.decode() if isinstance(item, bytes) else str(item) for item in array.reshape(-1)]
                axis_group.create_dataset(
                    "values", data=np.asarray(strings, dtype=_STRING_DTYPE).reshape(array.shape)
                )
            else:
                axis_group.create_dataset("values", data=array, **_dataset_options(array))
            for key in ("label", "unit", "description", "scale"):
                value = attrs.get(key)
                if value not in (None, ""):
                    axis_group.attrs[key] = value

        raw = h5.create_group("raw")
        raw_iq = getattr(result, "raw_iq", None)
        dataset_dims = getattr(result, "dataset_dims", {}) or {}
        if isinstance(raw_iq, dict):
            _write_tree(raw, _decorate_tree_with_dims(raw_iq, dataset_dims))
        elif raw_iq is not None:
            dims = dataset_dims.get("iq", [])
            _write_tree(raw, {"iq": {"values": raw_iq, "dims": dims}})
        _write_tree(
            raw,
            _decorate_tree_with_dims(getattr(result, "raw_data", {}) or {}, dataset_dims),
        )

        analysis = h5.create_group("analysis")
        _write_tree(analysis, getattr(result, "analysis_data", {}) or {})

        results = h5.create_group("results")
        _write_text(results, "fit_result_json", _json_dumps(getattr(result, "fit_result", {}) or {}))
        if getattr(result, "fit_params", None) is not None:
            _write_tree(results, {"fit_params": result.fit_params})
        if getattr(result, "fit_errors", None) is not None:
            _write_tree(results, {"fit_errors": result.fit_errors})
        summary = {
            "scalar_result": getattr(result, "scalar_result", None),
            "quality_message": getattr(result, "quality_message", ""),
            "avg_count": int(getattr(result, "avg_count", 0)),
        }
        _write_text(results, "summary_json", _json_dumps(summary))
        h5.flush()
        h5.attrs.modify("write_complete", True)
        h5.flush()


def save_result(
    result,
    path: Optional[os.PathLike | str] = None,
    *,
    comment: str = "",
    tags: Iterable[str] | str = (),
    data_root: Optional[os.PathLike | str] = None,
    catalog: bool = True,
) -> Path:
    """Atomically save one experiment and register it in the local catalog."""
    root = _resolve_data_root(data_root)
    explicit_path = path is not None
    utc_time = _as_utc(getattr(result, "timestamp", None))
    local_time = utc_time.astimezone(LOCAL_TIMEZONE)
    if not validate_experiment_id(getattr(result, "experiment_id", "")):
        result.experiment_id = generate_experiment_id(utc_time)
    id_catalog_root = (
        root
        if not explicit_path or data_root is not None
        else Path(path).expanduser().resolve().parent
    )
    while _experiment_id_exists(id_catalog_root, result.experiment_id):
        result.experiment_id = generate_experiment_id(utc_time)
    result.timestamp = utc_time
    result.comment = str(comment if comment != "" else getattr(result, "comment", ""))
    result.tags = _normalise_tags(tags if tags else getattr(result, "tags", []))

    if explicit_path:
        final_path = Path(path).expanduser().resolve()
        if final_path.suffix.lower() not in {".h5", ".hdf5"}:
            final_path = final_path.with_suffix(".h5")
        catalog_root = root if data_root is not None else final_path.parent
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment: {final_path}")
    else:
        catalog_root = root
        final_path = _auto_path(root, result, local_time, result.experiment_id)
        while final_path.exists():
            result.experiment_id = generate_experiment_id(utc_time)
            final_path = _auto_path(root, result, local_time, result.experiment_id)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()
    _write_file(partial_path, result, comment=result.comment, tags=result.tags, utc_time=utc_time)
    os.replace(partial_path, final_path)
    if catalog:
        _register_file(final_path, catalog_root)
    return final_path


def _attrs_dict(h5: h5py.File) -> dict:
    result = {}
    for key, value in h5.attrs.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result[key] = value
    return result


def inspect_file(path: os.PathLike | str) -> dict:
    """Read lightweight root/meta information without loading raw arrays."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as h5:
        attrs = _attrs_dict(h5)
        if attrs.get("schema_name") != SCHEMA_NAME:
            meta = _json_loads(h5.attrs.get("meta"), {}) or {}
            return {
                "path": str(path),
                "schema_name": "qickworkspace.legacy",
                "schema_version": "0",
                **meta,
            }
        meta_group = h5.get("meta")
        metadata = _json_loads(_read_text(meta_group, "metadata_json"), {}) if meta_group else {}
        config = _json_loads(_read_text(meta_group, "config_json"), {}) if meta_group else {}
        lineage = _json_loads(_read_text(meta_group, "lineage_json"), {}) if meta_group else {}
        tags = []
        comment = ""
        if meta_group is not None:
            comment = _read_text(meta_group, "comment")
            if "tags" in meta_group:
                tags = [item.decode() if isinstance(item, bytes) else str(item) for item in meta_group["tags"][:]]
        qubits = (
            (metadata or {}).get("qubit_names")
            or (metadata or {}).get("qubits")
            or (metadata or {}).get("qubit")
            or (config or {}).get("name")
            or (config or {}).get("qubit")
            or []
        )
        if isinstance(qubits, (str, int)):
            qubits = [qubits]
        return {
            "path": str(path),
            **attrs,
            "comment": comment,
            "tags": tags,
            "metadata": metadata or {},
            "config": config or {},
            "lineage": lineage or {},
            "qubits": [str(item) for item in qubits],
        }


def load_result(path: os.PathLike | str):
    """Load a native v1 file, with fallback support for the previous local format."""
    path = Path(path).expanduser().resolve()
    from ..core.experiment_data import ExperimentData, QualityFlag

    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_name") != SCHEMA_NAME:
            return _load_previous_format(h5)
        if not bool(h5.attrs.get("write_complete", False)):
            raise ValueError(f"Experiment file is incomplete: {path}")
        experiment_id = str(h5.attrs.get("experiment_id", ""))
        if not validate_experiment_id(experiment_id):
            raise ValueError(f"Invalid experiment_id in {path}: {experiment_id!r}")

        meta = h5["meta"]
        config = _json_loads(_read_text(meta, "config_json"), {}) or {}
        metadata = _json_loads(_read_text(meta, "metadata_json"), {}) or {}
        lineage = _json_loads(_read_text(meta, "lineage_json"), {}) or {}
        results = h5["results"]
        summary = _json_loads(_read_text(results, "summary_json"), {}) or {}
        fit_result = _json_loads(_read_text(results, "fit_result_json"), {}) or {}
        timestamp = datetime.fromisoformat(str(h5.attrs["timestamp_utc"]))
        quality_raw = str(h5.attrs.get("quality", "no_information"))
        try:
            quality = QualityFlag(quality_raw)
        except ValueError:
            quality = QualityFlag.NO_INFORMATION

        raw_tree = _read_tree(h5["raw"])
        raw_dims = _collect_dataset_dims(h5["raw"])
        if "iq" in raw_tree:
            raw_iq = raw_tree["iq"]
            raw_data = {key: value for key, value in raw_tree.items() if key != "iq"}
        else:
            raw_iq = raw_tree or None
            raw_data = {}
        analysis_data = _read_tree(h5["analysis"], preserve_attrs=True)
        axes = {}
        x_axis = y_axis = None
        for name, group in h5["axes"].items():
            values = group["values"][:]
            if values.dtype.kind in {"O", "S"}:
                values = np.asarray([
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in values.reshape(-1)
                ]).reshape(values.shape)
            payload = {"values": values}
            for key in ("label", "unit", "description", "scale"):
                if key in group.attrs:
                    payload[key] = group.attrs[key]
            axes[name] = payload
            if name == "x":
                scale = float(group.attrs.get("scale", 1.0)) or 1.0
                x_axis = values / scale
            elif name == "y":
                scale = float(group.attrs.get("scale", 1.0)) or 1.0
                y_axis = values / scale

        tags = [item.decode() if isinstance(item, bytes) else str(item) for item in meta["tags"][:]]
        obj = ExperimentData(
            experiment_type=str(h5.attrs.get("experiment_type", "")),
            experiment_id=experiment_id,
            timestamp=timestamp,
            raw_iq=raw_iq,
            x_axis=x_axis,
            y_axis=y_axis,
            fit_params=_read_tree(results["fit_params"]) if "fit_params" in results else None,
            fit_errors=_read_tree(results["fit_errors"]) if "fit_errors" in results else None,
            fit_result=fit_result,
            scalar_result=summary.get("scalar_result"),
            quality=quality,
            quality_message=summary.get("quality_message", ""),
            config=config,
            metadata=metadata,
            parent_id=lineage.get("parent_id"),
            children=lineage.get("children") or [],
            interrupted=bool(h5.attrs.get("interrupted", False)),
            avg_count=int(summary.get("avg_count", 0)),
            x_name=str(h5["axes/x"].attrs.get("label", "")) if "x" in h5["axes"] else "",
            x_unit=str(h5["axes/x"].attrs.get("unit", "")) if "x" in h5["axes"] else "",
            x_scale=float(h5["axes/x"].attrs.get("scale", 1.0)) if "x" in h5["axes"] else 1.0,
            y_name=str(h5["axes/y"].attrs.get("label", "")) if "y" in h5["axes"] else "",
            y_unit=str(h5["axes/y"].attrs.get("unit", "")) if "y" in h5["axes"] else "",
            y_scale=float(h5["axes/y"].attrs.get("scale", 1.0)) if "y" in h5["axes"] else 1.0,
            axes=axes,
            raw_data=raw_data,
            analysis_data=analysis_data,
            dataset_dims=raw_dims,
            data_kind=str(h5.attrs.get("data_kind", "")),
            analysis_id=str(h5.attrs.get("analysis_id", "")),
            plot_id=str(h5.attrs.get("plot_id", "")),
            comment=_read_text(meta, "comment"),
            tags=tags,
            session_id=lineage.get("session_id"),
        )
        return obj


def _load_previous_format(h5: h5py.File):
    from ..core.experiment_data import ExperimentData

    meta = _json_loads(h5.attrs.get("meta"), {}) or {}
    obj = ExperimentData.from_dict(meta)
    if "data" in h5:
        data = h5["data"]
        obj.raw_iq = data["avgi"][:] + 1j * data["avgq"][:]
    if "x" in h5:
        scale = obj.x_scale or 1.0
        obj.x_axis = h5["x/values"][:] / scale
    if "y" in h5:
        scale = obj.y_scale or 1.0
        obj.y_axis = h5["y/values"][:] / scale
    return obj


@dataclass
class ValidationReport:
    path: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_file(path: os.PathLike | str) -> ValidationReport:
    """Validate schema identity, completion state, ID, and dataset dimensions."""
    errors: list[str] = []
    warnings: list[str] = []
    resolved = str(Path(path).expanduser().resolve())
    try:
        with h5py.File(resolved, "r") as h5:
            if h5.attrs.get("schema_name") != SCHEMA_NAME:
                errors.append("schema_name is not qickworkspace.experiment")
            if str(h5.attrs.get("schema_version", "")) != SCHEMA_VERSION:
                errors.append(f"unsupported schema_version {h5.attrs.get('schema_version')!r}")
            if not bool(h5.attrs.get("write_complete", False)):
                errors.append("write_complete is false")
            if not validate_experiment_id(str(h5.attrs.get("experiment_id", ""))):
                errors.append("experiment_id is invalid")
            for group in ("meta", "axes", "raw", "analysis", "results"):
                if group not in h5:
                    errors.append(f"missing group: {group}")
            if "raw" in h5 and len(h5["raw"]) == 0:
                warnings.append("raw group contains no datasets")
            axis_lengths = {
                name: int(group["values"].size)
                for name, group in h5.get("axes", {}).items()
                if "values" in group
            }
            for group_name in ("raw", "analysis"):
                if group_name not in h5:
                    continue
                def _check_dims(name, node):
                    if not isinstance(node, h5py.Dataset) or "dims" not in node.attrs:
                        return
                    dims = list(_json_loads(node.attrs["dims"], []) or [])
                    if len(dims) != node.ndim:
                        errors.append(
                            f"{group_name}/{name}: {len(dims)} dims for rank-{node.ndim} dataset"
                        )
                        return
                    for index, dim in enumerate(dims):
                        if dim not in axis_lengths:
                            errors.append(f"{group_name}/{name}: missing axis {dim!r}")
                        elif node.shape[index] != axis_lengths[dim]:
                            errors.append(
                                f"{group_name}/{name}: shape[{index}]={node.shape[index]} "
                                f"does not match axis {dim!r} length {axis_lengths[dim]}"
                            )
                h5[group_name].visititems(_check_dims)
    except Exception as exc:
        errors.append(str(exc))
    return ValidationReport(resolved, not errors, errors, warnings)


def _catalog_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / CATALOG_FILENAME


def _catalog_contains_id(root: Path, experiment_id: str) -> bool:
    catalog = _catalog_path(root)
    if not catalog.exists():
        return False
    try:
        with sqlite3.connect(catalog, timeout=5.0) as connection:
            row = connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ? LIMIT 1",
                (experiment_id,),
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _experiment_id_exists(root: Path, experiment_id: str) -> bool:
    if _catalog_path(root).exists():
        return _catalog_contains_id(root, experiment_id)
    if not root.exists():
        return False
    for path in root.rglob("*.h5"):
        try:
            with h5py.File(path, "r") as h5:
                if str(h5.attrs.get("experiment_id", "")) == experiment_id:
                    return True
        except OSError:
            continue
    return False


def _connect_catalog(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_catalog_path(root), timeout=30.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            timestamp_local TEXT NOT NULL,
            experiment_type TEXT NOT NULL,
            qubits_json TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            quality TEXT NOT NULL,
            session_id TEXT,
            comment_preview TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            schema_version TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_experiments_time ON experiments(timestamp_utc)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(experiment_type)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_experiments_session ON experiments(session_id)")
    return connection


def _register_file(path: Path, root: Path) -> None:
    info = inspect_file(path)
    metadata = info.get("metadata", {}) or {}
    qubits = info.get("qubits") or metadata.get("qubit_names") or []
    if isinstance(qubits, (str, int)):
        qubits = [qubits]
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        relative = os.path.relpath(path.resolve(), root.resolve())
    lineage = info.get("lineage", {}) or {}
    row = (
        info.get("experiment_id", ""), info.get("timestamp_utc", ""),
        info.get("timestamp_local", ""), info.get("experiment_type", ""),
        _json_dumps([str(q) for q in qubits]), _json_dumps(info.get("tags", [])),
        info.get("quality", "no_information"), lineage.get("session_id"),
        str(info.get("comment", ""))[:240], relative, info.get("schema_version", ""),
    )
    with _connect_catalog(root) as connection:
        connection.execute(
            """INSERT OR REPLACE INTO experiments
            (experiment_id,timestamp_utc,timestamp_local,experiment_type,qubits_json,
             tags_json,quality,session_id,comment_preview,relative_path,schema_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            row,
        )


@dataclass(frozen=True)
class ExperimentReference:
    experiment_id: str
    timestamp_utc: str
    timestamp_local: str
    experiment_type: str
    qubits: tuple[str, ...]
    tags: tuple[str, ...]
    quality: str
    session_id: Optional[str]
    comment_preview: str
    path: Path

    def load(self):
        return load_result(self.path)


def _date_boundary(value: Any, *, end: bool = False) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, date):
        local = datetime.combine(value, time.max if end else time.min, tzinfo=LOCAL_TIMEZONE)
        return local.astimezone(timezone.utc).isoformat()
    text = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        parsed_date = date.fromisoformat(text)
        parsed = datetime.combine(parsed_date, time.max if end else time.min, tzinfo=LOCAL_TIMEZONE)
    else:
        parsed = datetime.fromisoformat(text)
    return _as_utc(parsed).isoformat()


def find_experiments(
    experiment_type: Optional[str] = None,
    qubit: Optional[str] = None,
    tags: Optional[Iterable[str] | str] = None,
    quality: Optional[str] = None,
    start: Any = None,
    end: Any = None,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
    *,
    data_root: Optional[os.PathLike | str] = None,
) -> list[ExperimentReference]:
    """Search the rebuildable catalog and return lazy experiment references."""
    root = _resolve_data_root(data_root)
    if not _catalog_path(root).exists():
        rebuild_catalog(root)
    conditions = []
    params: list[Any] = []
    for column, value in (
        ("experiment_type", experiment_type), ("quality", quality), ("session_id", session_id)
    ):
        if value is not None:
            conditions.append(f"{column} = ?")
            params.append(str(value))
    if start is not None:
        conditions.append("timestamp_utc >= ?")
        params.append(_date_boundary(start))
    if end is not None:
        conditions.append("timestamp_utc <= ?")
        params.append(_date_boundary(end, end=True))
    query = "SELECT * FROM experiments"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY timestamp_utc DESC"
    requested_tags = set(_normalise_tags(tags))
    references = []
    with _connect_catalog(root) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query, params):
            row_tags = tuple(_json_loads(row["tags_json"], []) or [])
            row_qubits = tuple(str(q) for q in (_json_loads(row["qubits_json"], []) or []))
            if qubit is not None and str(qubit) not in row_qubits:
                continue
            if requested_tags and not requested_tags.issubset(row_tags):
                continue
            path = (root / row["relative_path"]).resolve()
            if not path.exists():
                continue
            references.append(ExperimentReference(
                experiment_id=row["experiment_id"], timestamp_utc=row["timestamp_utc"],
                timestamp_local=row["timestamp_local"], experiment_type=row["experiment_type"],
                qubits=row_qubits, tags=row_tags, quality=row["quality"],
                session_id=row["session_id"], comment_preview=row["comment_preview"], path=path,
            ))
            if limit is not None and len(references) >= int(limit):
                break
    return references


def rebuild_catalog(data_root: os.PathLike | str) -> int:
    """Recreate the SQLite catalog from completed native HDF5 files."""
    root = _resolve_data_root(data_root)
    catalog = _catalog_path(root)
    if catalog.exists():
        catalog.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(catalog) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    count = 0
    for path in sorted(root.rglob("*.h5")):
        try:
            info = inspect_file(path)
            if info.get("schema_name") != SCHEMA_NAME or not info.get("write_complete"):
                continue
            _register_file(path, root)
            count += 1
        except Exception:
            continue
    if not catalog.exists():
        with _connect_catalog(root):
            pass
    return count


def convert_labber_file(
    source: os.PathLike | str,
    destination: Optional[os.PathLike | str] = None,
    *,
    metadata: Optional[dict] = None,
) -> Path:
    """Best-effort conversion isolated behind the optional Labber dependency."""
    try:
        import Labber
    except ImportError as exc:
        raise ImportError(
            "Labber is only required for convert_labber_file(); install it in the Labber environment."
        ) from exc
    from ..core.experiment_data import ExperimentData

    log = Labber.LogFile(str(source))
    log_channels = log.getLogChannels()
    if not log_channels:
        raise ValueError(f"No Labber log channels found in {source}")
    channel_name = log_channels[0]["name"] if isinstance(log_channels[0], dict) else str(log_channels[0])
    raw = np.asarray(log.getData(channel_name))
    step_channels = log.getStepChannels() or []
    axes = {}
    for index, channel in enumerate(step_channels):
        if not isinstance(channel, dict):
            continue
        values = channel.get("values")
        if values is not None:
            axes[f"axis_{index}"] = {
                "values": np.asarray(values),
                "label": channel.get("name", f"axis_{index}"),
                "unit": channel.get("unit", ""),
            }
    supplied = dict(metadata or {})
    result = ExperimentData(
        experiment_type=supplied.pop("experiment_type", Path(source).stem),
        raw_iq=raw,
        metadata={"legacy_source": str(source), "legacy_unknown": True, **supplied},
        axes=axes,
        data_kind="legacy_unknown",
        analysis_id="legacy_unknown",
        plot_id="legacy_unknown",
        comment=str(getattr(log, "getComment", lambda: "")() or ""),
    )
    return save_result(result, destination)


__all__ = [
    "SCHEMA_NAME", "SCHEMA_VERSION", "ExperimentReference", "ValidationReport",
    "generate_experiment_id", "validate_experiment_id", "save_result", "load_result",
    "inspect_file", "validate_file", "find_experiments", "rebuild_catalog",
    "convert_labber_file",
]
