"""NVIDIA QCA Blueprint AST-compatible wrappers and tagged JSON result payloads."""

from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from QickworkspaceV2.data.serialization import jsonable


def to_blueprint(result, *, artifact_dir=None, max_array_values=20000):
    """Provide actual arrays + PNG, not merely paths outside Blueprint's storage."""
    data = {}
    array_metadata, omitted_arrays = {}, {}
    remaining = max_array_values
    for q, fit in result.fits.items():
        for key, value in fit.parameters.items():
            data[f"{q}_{key}"] = {
                "type": "scalar",
                "value": value,
                "unit": fit.units.get(key, ""),
                "error": fit.errors.get(key),
                "quality": "accepted" if fit.success else "rejected",
            }
    for q, trace in result.traces.items():
        arrays = [
            ("I", trace.iq.real, trace.dims, "ADC"),
            ("Q", trace.iq.imag, trace.dims, "ADC"),
            *((axis, values, (axis,), trace.units.get(axis, "")) for axis, values in trace.coords.items()),
        ]
        if trace.shots is not None:
            arrays.extend(
                [("shots_I", trace.shots.real, trace.shot_dims, "ADC"),
                 ("shots_Q", trace.shots.imag, trace.shot_dims, "ADC")]
            )
        for label, values, dims, unit in arrays:
            values = np.asarray(values)
            name = f"{q}_{label}"
            descriptor = {"dims": list(dims), "shape": list(values.shape), "unit": unit}
            array_metadata[name] = {**descriptor, "dtype": str(values.dtype)}
            reason = "array_export_limit" if values.size > remaining else None
            if reason is None and values.dtype.kind not in "biuf":
                reason = "non_numeric_dtype"
                # HDF5's numeric array contract cannot store Unicode coordinates.
                # Retain bounded state/readout labels in the portable metadata.
                if values.dtype.kind in "US" or (
                    values.dtype.kind == "O" and all(isinstance(value, (str, bytes)) for value in values.flat)
                ):
                    array_metadata[name]["labels"] = values.astype(str).tolist()
                    remaining -= values.size
            if reason is not None:
                omitted_arrays[name] = {**descriptor, "reason": reason, "required_values": int(values.size)}
                continue
            data[name] = {
                "type": "array",
                "value": values.tolist(),
                **descriptor,
            }
            remaining -= values.size
    plots = []
    try:
        figure = result.plot()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=120)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        plots = [{"name": result.experiment, "format": "png", "data": encoded}]
        if artifact_dir is not None:
            (Path(artifact_dir) / "plot.png").write_bytes(buffer.getvalue())
    except Exception as exc:
        plot_error = str(exc)
    else:
        plot_error = ""
    accepted = result.analysis_status != "failed" and (not result.fits or result.is_good())
    payload = {
        "status": "success" if accepted else "failed",
        "data": data,
        "plots": plots,
        "metadata": {
            "run_id": result.run_id,
            "experiment": result.experiment,
            "targets": result.metadata["targets"],
            "backend": "qick",
            "analysis_status": result.analysis_status,
            "analysis_message": result.analysis_message,
            "plot_error": plot_error,
            "data_path": str(result.path) if result.path else None,
            "array_export_limit": max_array_values,
            "array_metadata": array_metadata,
            "omitted_arrays": omitted_arrays,
            "calibration_updated": False,
        },
    }
    if not accepted:
        payload["error"] = result.analysis_message or "Fit quality checks failed"
    return jsonable(payload)


def execute(experiment, parameters, *, worker_url=None, timeout=240, request_id=None):
    import os
    from QickworkspaceV2.integrations.client import WorkerClient

    client = WorkerClient(worker_url or os.environ.get("QICK_WORKER_URL", "http://127.0.0.1:8000"))
    try:
        return client.run(experiment, parameters=parameters, timeout=timeout, request_id=request_id)
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "metadata": {"experiment": experiment}}
    finally:
        client.close()


def export_wrappers(registry, directory):
    """One public, typed function per file; no decorator-dependent discovery."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "__init__.py").touch()
    paths = []
    exported_names = set()
    for entry in registry.catalog():
        import keyword

        function = re.sub(r"[^A-Za-z0-9_]", "_", entry["id"])
        if (
            not function.isidentifier()
            or function.startswith("_")
            or keyword.iskeyword(function)
            or function in exported_names
        ):
            raise ValueError(f"Cannot export experiment ID {entry['id']}")
        exported_names.add(function)
        schema = entry["parameters"]
        arguments = []
        for name, info in schema.get("properties", {}).items():
            if not name.isidentifier():
                raise ValueError(f"Invalid parameter name {name}")
            primitive = {"string": "str", "number": "float", "integer": "int", "boolean": "bool"}.get(
                info.get("type")
            )
            if primitive is None:
                raise ValueError(
                    f"{entry['id']}/{name}: Blueprint wrapper needs scalar parameters; add a scalar adapter"
                )
            if primitive in ("int", "float"):
                minimum = info.get("minimum", info.get("exclusiveMinimum"))
                maximum = info.get("maximum", info.get("exclusiveMaximum"))
                if minimum is not None and maximum is not None:
                    primitive = f"Annotated[{primitive}, ({minimum!r}, {maximum!r})]"
            default = f" = {info['default']!r}" if "default" in info else ""
            arguments.append(("default" in info, name, f"{name}: {primitive}{default}"))
        arguments.sort(key=lambda x: x[0])
        if any(a[1] == "request_id" for a in arguments):
            raise ValueError("request_id is reserved for worker idempotency")
        signature = ", ".join([*(a[2] for a in arguments), 'request_id: str = ""'])
        values = ", ".join(f"{name!r}: {name}" for _, name, _ in arguments)
        description = entry["description"].replace('"""', "'''")
        imports = "from typing import Annotated\n\n\n" if "Annotated[" in signature else ""
        source = (
            "# Generated from ExperimentRegistry; edit the experiment module and re-export.\n"
            + imports
            + f"def {function}({signature}) -> dict:\n"
            + f'    """{description}. Reuse request_id when retrying the same measurement."""\n'
            + "    from QickworkspaceV2.integrations.nvidia import execute\n"
            + f"    return execute({entry['id']!r}, {{{values}}}, request_id=request_id or None)\n"
        )
        path = directory / f"{function}.py"
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    return paths
