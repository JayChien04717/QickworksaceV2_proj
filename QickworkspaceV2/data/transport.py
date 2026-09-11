"""Portable worker results with typed arrays, labels and PNG plots."""

from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from QickworkspaceV2.data.serialization import jsonable


def to_worker_result(result, *, artifact_dir=None, max_array_values=20000):
    """Serialize acquired arrays, fit quality and plots for worker clients."""
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
