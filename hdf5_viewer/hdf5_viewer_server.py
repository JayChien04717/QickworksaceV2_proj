"""Local-only web server for browsing Qickworkspace SQLite/HDF5 archives."""

from __future__ import annotations

import json
import sqlite3
import sys
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
    inspect_file,
    load_result,
    rebuild_catalog,
    validate_file,
    _native_root,
)


HTML_FILE = APP_DIR / "hdf5_viewer.html"
app = FastAPI(title="Qickworkspace HDF5 Viewer", docs_url=None, redoc_url=None)


def _root(folder: str) -> Path:
    if not str(folder).strip():
        raise HTTPException(400, "Folder path is required")
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(404, f"Folder does not exist: {root}")
    return root


def _catalog(root: Path, *, build: bool = False) -> Path:
    catalog = root / CATALOG_FILENAME
    if not catalog.exists() and build:
        rebuild_catalog(root)
    if not catalog.is_file():
        raise HTTPException(
            404,
            f"No {CATALOG_FILENAME} in {root}. Save a hybrid/native experiment first or choose Rebuild.",
        )
    return catalog


def _rows(root: Path, *, limit: int = 2000):
    catalog = _catalog(root)
    with sqlite3.connect(f"file:{catalog.as_posix()}?mode=ro", uri=True, timeout=10) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY timestamp_utc DESC LIMIT ?", (int(limit),)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["qubits"] = json.loads(item.pop("qubits_json") or "[]")
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        item["path"] = str((root / item["relative_path"]).resolve())
        result.append(item)
    return result


def _row(root: Path, experiment_id: str):
    catalog = _catalog(root)
    with sqlite3.connect(f"file:{catalog.as_posix()}?mode=ro", uri=True, timeout=10) as connection:
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


def _dataset_keys(path: Path, section: str) -> list[str]:
    keys = []
    with h5py.File(path, "r") as h5:
        root = _native_root(h5)
        if root is None or section not in root:
            return keys
        root[section].visititems(
            lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None
        )
    return keys


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
            first = values.reshape(-1)[:5]
            summary[name] = {
                "size": int(values.size),
                "shape": list(values.shape),
                "unit": str(axis.attrs.get("unit", "")),
                "label": str(axis.attrs.get("label", "")),
                "first": _jsonable(first),
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
        values = np.asarray(node)
        dims_value = node.attrs.get("dims", "[]")
        if isinstance(dims_value, bytes):
            dims_value = dims_value.decode("utf-8")
        try:
            dims = json.loads(dims_value) if isinstance(dims_value, str) else list(dims_value)
        except (TypeError, json.JSONDecodeError):
            dims = []
        return values, dims

@app.get("/")
def index():
    return FileResponse(HTML_FILE)


@app.get("/api/catalog")
def catalog(folder: str = Query(...), limit: int = Query(2000, ge=1, le=10000)):
    root = _root(folder)
    rows = _rows(root, limit=limit)
    return {
        "folder": str(root),
        "catalog": str(root / CATALOG_FILENAME),
        "count": len(rows),
        "experiments": rows,
    }


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
    axis_summary = _axis_summary(path)
    return JSONResponse(
        _jsonable(
            {
                **info,
                "valid": report.valid,
                "validation_errors": report.errors,
                "validation_warnings": report.warnings,
                "raw_keys": _dataset_keys(path, "raw"),
                "analysis_keys": _dataset_keys(path, "analysis"),
                "axes": axis_summary,
            }
        )
    )


@app.get("/api/data/{experiment_id}")
def data(
    experiment_id: str,
    folder: str = Query(...),
    dataset: str = Query("iq"),
    source: str = Query("raw", pattern="^(raw|analysis)$"),
    max_points: int = Query(4000, ge=100, le=50000),
):
    root = _root(folder)
    _, path = _row(root, experiment_id)
    try:
        array, dims = _read_dataset(path, source, dataset)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    flat = array.reshape(-1)
    stride = max(1, int(np.ceil(flat.size / max_points)))
    sampled = flat[::stride]
    payload = {
        "dataset": f"{source}/{dataset}",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dims": list(dims),
        "stride": stride,
        "sample_count": int(sampled.size),
    }
    if np.iscomplexobj(sampled):
        payload.update(
            real=sampled.real.tolist(),
            imag=sampled.imag.tolist(),
            magnitude=np.abs(sampled).tolist(),
        )
    elif np.issubdtype(sampled.dtype, np.number):
        payload["values"] = sampled.tolist()
    else:
        payload["values"] = sampled.astype(str).tolist()
    return JSONResponse(payload)


@app.get("/api/plot/{experiment_id}")
def plot(
    experiment_id: str,
    folder: str = Query(...),
    name: str = Query("main.png", pattern="^(main|analysis|preview)\\.png$"),
):
    root = _root(folder)
    _, path = _row(root, experiment_id)
    with h5py.File(path, "r") as h5:
        plot_candidates = []
        for group_name in (EMBEDDED_GROUP, *LEGACY_EMBEDDED_GROUPS):
            plot_candidates.append(f"{group_name}/plots/{name}")
        for group_name in (EMBEDDED_GROUP, *LEGACY_EMBEDDED_GROUPS):
            plot_candidates.extend([
                f"{group_name}/plots/main.png",
                f"{group_name}/plots/analysis.png",
                f"{group_name}/plots/preview.png",
            ])
        for embedded in plot_candidates:
            if embedded in h5:
                return Response(np.asarray(h5[embedded], dtype=np.uint8).tobytes(), media_type="image/png")

    # Standalone native files get the same hardware-independent preview.
    from QickworkspaceV2.tools.Labber_saver import _preview_png

    return Response(_preview_png(load_result(path)), media_type="image/png")


if __name__ == "__main__":
    import threading
    import webbrowser
    import uvicorn

    url = "http://127.0.0.1:8765"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Opening {url}")
    uvicorn.run(app, host="127.0.0.1", port=8765)
