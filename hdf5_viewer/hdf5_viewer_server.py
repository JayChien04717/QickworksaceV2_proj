"""Local-only web server for browsing, plotting, and comparing Qickworkspace HDF5 archives."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QickworkspaceV2.tools.hdf5_store import (
    CATALOG_FILENAME,
    EMBEDDED_GROUP,
    LEGACY_EMBEDDED_GROUPS,
    _native_root,
    _normalise_tags,
    inspect_file,
    load_result,
    rebuild_catalog,
    validate_file,
)

HTML_FILE = APP_DIR / "hdf5_viewer.html"
app = FastAPI(title="Qickworkspace HDF5 Viewer", docs_url=None, redoc_url=None)
_DIALOG_LOCK = threading.Lock()


def _root(folder: str) -> Path:
    if not str(folder).strip():
        raise HTTPException(400, "Folder path is required")
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, f"Folder does not exist: {root}")
    return root


def _catalog(root: Path) -> Path:
    catalog = root / CATALOG_FILENAME
    if not catalog.is_file():
        raise HTTPException(404, f"No {CATALOG_FILENAME} in {root}. Choose Rebuild catalog first.")
    return catalog


def _rows(root: Path, *, limit: int = 2000):
    catalog = _catalog(root)
    with sqlite3.connect(f"file:{catalog.as_posix()}?mode=ro&immutable=1", uri=True, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY timestamp_utc DESC LIMIT ?", (int(limit),)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["qubits"] = json.loads(item.pop("qubits_json") or "[]")
        item["tags"] = _normalise_tags(
            json.loads(item.pop("tags_json") or "[]"),
            item.get("experiment_type", ""),
        )
        item["path"] = str((root / item["relative_path"]).resolve())
        result.append(item)
    return result


def _row(root: Path, experiment_id: str):
    with sqlite3.connect(f"file:{_catalog(root).as_posix()}?mode=ro&immutable=1", uri=True, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"Experiment not found: {experiment_id}")
    path = (root / row["relative_path"]).resolve()
    if not path.is_file():
        raise HTTPException(404, f"Indexed HDF5 file is missing: {path}")
    return dict(row), path


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def _decode_dims(node: h5py.Dataset) -> list[str]:
    value = node.attrs.get("dims", "[]")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return [str(item) for item in (json.loads(value) if isinstance(value, str) else value)]
    except (TypeError, json.JSONDecodeError):
        return []


def _dataset_descriptor(name: str, node: h5py.Dataset) -> dict:
    dtype = np.dtype(node.dtype)
    return {
        "name": name,
        "shape": list(node.shape),
        "size": int(node.size),
        "dtype": str(dtype),
        "dims": _decode_dims(node),
        "complex": bool(np.issubdtype(dtype, np.complexfloating)),
        "numeric": bool(np.issubdtype(dtype, np.number)),
        "rank": int(node.ndim),
    }


def _logical_datasets(path: Path, section: str) -> list[dict]:
    result: list[dict] = []
    with h5py.File(path, "r") as h5:
        root = _native_root(h5)
        if root is None or section not in root:
            return result

        def visit(group: h5py.Group, prefix: str = "") -> None:
            for name, node in group.items():
                logical_name = f"{prefix}/{name}" if prefix else name
                if isinstance(node, h5py.Group) and "values" in node and isinstance(node["values"], h5py.Dataset):
                    result.append(_dataset_descriptor(logical_name, node["values"]))
                elif isinstance(node, h5py.Group):
                    visit(node, logical_name)
                elif isinstance(node, h5py.Dataset):
                    result.append(_dataset_descriptor(logical_name, node))

        visit(root[section])
    return result


def _axis_summary(path: Path) -> dict:
    summary = {}
    with h5py.File(path, "r") as h5:
        root = _native_root(h5)
        if root is None or "axes" not in root:
            return summary
        for name, axis in root["axes"].items():
            if not isinstance(axis, h5py.Group) or "values" not in axis:
                continue
            values = np.asarray(axis["values"])
            numeric = np.issubdtype(values.dtype, np.number)
            summary[name] = {
                "size": int(values.size), "shape": list(values.shape),
                "unit": str(axis.attrs.get("unit", "")), "label": str(axis.attrs.get("label", "")),
                "min": _jsonable(np.nanmin(values)) if numeric and values.size else None,
                "max": _jsonable(np.nanmax(values)) if numeric and values.size else None,
            }
    return summary


def _read_dataset(path: Path, section: str, dataset: str):
    with h5py.File(path, "r") as h5:
        root = _native_root(h5)
        if root is None or section not in root:
            raise KeyError(f"Missing section: {section}")
        node = root[section].get(dataset)
        if node is None:
            raise KeyError(f"Missing dataset: {section}/{dataset}")
        if isinstance(node, h5py.Group) and "values" in node:
            node = node["values"]
        if not isinstance(node, h5py.Dataset):
            raise KeyError(f"Not a dataset: {section}/{dataset}")
        return np.asarray(node), _decode_dims(node)


def _read_axis(path: Path, name: str) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as h5:
        root = _native_root(h5)
        if root is None or f"axes/{name}/values" not in root:
            raise KeyError(name)
        group = root[f"axes/{name}"]
        return np.asarray(group["values"]), {
            key: _jsonable(group.attrs.get(key, "")) for key in ("label", "unit", "description", "scale")
        }


def _numeric_channel(values: np.ndarray, channel: str) -> tuple[np.ndarray, str]:
    channel = {"amplitude": "abs", "amp": "abs", "i": "real", "q": "imag"}.get(
        (channel or "auto").lower(), (channel or "auto").lower()
    )
    if np.iscomplexobj(values):
        channel = "abs" if channel in {"auto", "value"} else channel
        converters = {
            "abs": np.abs, "real": np.real, "imag": np.imag,
            "phase": lambda value: np.unwrap(np.angle(value), axis=-1),
        }
        if channel not in converters:
            raise ValueError(f"Unsupported complex channel: {channel}")
        return np.asarray(converters[channel](values), dtype=float), channel
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Selected dataset is not numeric")
    return np.asarray(values, dtype=float), "value"


def _one_dimensional(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1)
    if values.ndim == 1:
        return values
    return np.nanmean(values.reshape(-1, values.shape[-1]), axis=0)


def _sample_indices(size: int, max_points: int) -> np.ndarray:
    if size <= max_points:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, max_points, dtype=int))


def _result_summary(path: Path) -> dict:
    """Read fit summaries without materialising the potentially large raw tree."""
    def read_node(node):
        if isinstance(node, h5py.Group):
            return {name: read_node(child) for name, child in node.items()}
        value = node[()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return _jsonable(value)

    with h5py.File(path, "r") as h5:
        root = _native_root(h5)
        if root is None or "results" not in root:
            return {"fit_result": {}, "fit_params": None, "fit_errors": None,
                    "scalar_result": None, "quality_message": ""}
        results = root["results"]
        fit_text = read_node(results["fit_result_json"]) if "fit_result_json" in results else "{}"
        summary_text = read_node(results["summary_json"]) if "summary_json" in results else "{}"
        try:
            fit_result = json.loads(fit_text) if isinstance(fit_text, str) else {}
        except json.JSONDecodeError:
            fit_result = {}
        try:
            summary = json.loads(summary_text) if isinstance(summary_text, str) else {}
        except json.JSONDecodeError:
            summary = {}
        return {
            "fit_result": fit_result,
            "fit_params": read_node(results["fit_params"]) if "fit_params" in results else None,
            "fit_errors": read_node(results["fit_errors"]) if "fit_errors" in results else None,
            "scalar_result": summary.get("scalar_result"),
            "quality_message": summary.get("quality_message", ""),
        }


@app.get("/")
def index():
    return FileResponse(HTML_FILE)


@app.get("/api/choose-folder")
def choose_folder(initial: str = Query("")):
    """Open a native local folder picker; this viewer is intentionally local-only."""
    with _DIALOG_LOCK:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initial_dir = initial if initial and Path(initial).is_dir() else str(Path.home())
            selected = filedialog.askdirectory(parent=root, initialdir=initial_dir, mustexist=True)
            root.destroy()
        except Exception as exc:
            raise HTTPException(500, f"Could not open the native folder dialog: {exc}") from exc
    return {"folder": selected or ""}


@app.get("/api/catalog")
def catalog(folder: str = Query(...), limit: int = Query(2000, ge=1, le=10000)):
    root = _root(folder)
    rows = _rows(root, limit=limit)
    return {"folder": str(root), "catalog": str(root / CATALOG_FILENAME), "count": len(rows), "experiments": rows}


@app.post("/api/rebuild")
def rebuild(folder: str = Query(...)):
    root = _root(folder)
    try:
        count = rebuild_catalog(root)
    except PermissionError as exc:
        raise HTTPException(409, f"Catalog is open in another process: {exc}") from exc
    return {"folder": str(root), "indexed": count}


@app.get("/api/experiment/{experiment_id}")
def experiment(experiment_id: str, folder: str = Query(...)):
    root = _root(folder)
    _, path = _row(root, experiment_id)
    info = inspect_file(path)
    report = validate_file(path)
    return JSONResponse(_jsonable({
        **info, **_result_summary(path), "valid": report.valid,
        "validation_errors": report.errors, "validation_warnings": report.warnings,
        "datasets": {"raw": _logical_datasets(path, "raw"), "analysis": _logical_datasets(path, "analysis")},
        "axes": _axis_summary(path),
    }))


@app.get("/api/trace/{experiment_id}")
def trace(
    experiment_id: str,
    folder: str = Query(...),
    dataset: str = Query("iq"),
    source: str = Query("raw", pattern="^(raw|analysis)$"),
    channel: str = Query("auto", pattern="^(auto|value|abs|real|imag|phase)$"),
    max_points: int = Query(3000, ge=100, le=20000),
):
    root = _root(folder)
    _, path = _row(root, experiment_id)
    try:
        values, dims = _read_dataset(path, source, dataset)
        display_values, actual_channel = _numeric_channel(values, channel)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    y = _one_dimensional(display_values)
    axis_name = dims[-1] if dims else "x"
    axis_meta = {"label": axis_name, "unit": "", "scale": 1.0}
    try:
        x, axis_meta = _read_axis(path, axis_name)
        x = np.asarray(x).reshape(-1)
    except KeyError:
        x = np.arange(y.size, dtype=float)
    if x.size != y.size:
        x = np.arange(y.size, dtype=float)
        axis_meta = {"label": "index", "unit": "", "scale": 1.0}
    fit = None
    try:
        fit_values, fit_dims = _read_dataset(path, "analysis", "fit_curve")
        fit_values, _ = _numeric_channel(np.asarray(fit_values), actual_channel)
        fit = _one_dimensional(fit_values)
        if fit.size != y.size or (fit_dims and dims and fit_dims[-1] != dims[-1]):
            fit = None
    except (KeyError, ValueError, TypeError):
        pass
    indices = _sample_indices(y.size, max_points)
    result = _result_summary(path)
    metadata = inspect_file(path).get("metadata", {}) or {}
    return JSONResponse(_jsonable({
        "experiment_id": experiment_id, "experiment_type": inspect_file(path).get("experiment_type", ""),
        "dataset": f"{source}/{dataset}", "channel": actual_channel,
        "shape": list(values.shape), "dims": dims, "reduction": "mean leading dimensions" if values.ndim > 1 else "none",
        "x": x[indices], "y": y[indices], "fit": fit[indices] if fit is not None else None,
        "x_label": axis_meta.get("label") or axis_name, "x_unit": axis_meta.get("unit", ""),
        "y_label": metadata.get("fit_channel", actual_channel), **result,
    }))


@app.get("/api/data/{experiment_id}")
def data(
    experiment_id: str, folder: str = Query(...), dataset: str = Query("iq"),
    source: str = Query("raw", pattern="^(raw|analysis)$"), max_points: int = Query(4000, ge=100, le=50000),
):
    root = _root(folder)
    _, path = _row(root, experiment_id)
    try:
        array, dims = _read_dataset(path, source, dataset)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    flat = array.reshape(-1)
    indices = _sample_indices(flat.size, max_points)
    sampled = flat[indices]
    payload = {"dataset": f"{source}/{dataset}", "shape": list(array.shape), "dtype": str(array.dtype), "dims": dims, "sample_count": int(sampled.size)}
    if np.iscomplexobj(sampled):
        payload.update(real=sampled.real.tolist(), imag=sampled.imag.tolist(), magnitude=np.abs(sampled).tolist())
    elif np.issubdtype(sampled.dtype, np.number):
        payload["values"] = sampled.tolist()
    else:
        payload["values"] = sampled.astype(str).tolist()
    return JSONResponse(payload)


@app.get("/api/plot/{experiment_id}")
def plot(experiment_id: str, folder: str = Query(...), name: str = Query("main.png", pattern="^(main|analysis|preview)\\.png$")):
    root = _root(folder)
    _, path = _row(root, experiment_id)
    with h5py.File(path, "r") as h5:
        exact = [f"{group}/plots/{name}" for group in (EMBEDDED_GROUP, *LEGACY_EMBEDDED_GROUPS)]
        for embedded in exact:
            if embedded in h5:
                return Response(np.asarray(h5[embedded], dtype=np.uint8).tobytes(), media_type="image/png")
    from QickworkspaceV2.tools.Labber_saver import _preview_png
    return Response(_preview_png(load_result(path)), media_type="image/png")


if __name__ == "__main__":
    import webbrowser
    import uvicorn
    url = "http://127.0.0.1:8765"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Opening {url}")
    uvicorn.run(app, host="127.0.0.1", port=8765)
