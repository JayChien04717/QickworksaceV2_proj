"""Export V2 results to Labber logs while retaining the complete native record."""
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os

import h5py
import numpy as np

from .format import create_file
from labtools.hdf5 import write_record, result_record
from labtools.catalog import register_file_safely
from labtools.serialization import dumps


def save_labber_results(results, *, load=None, **options):
    """Export every result/target in a Notebook procedure, including scan children.

    Accepts result objects, lists, procedure dictionaries and run-ID table rows.
    load is the SDK store loader for child runs and table rows. Scalar procedure
    settings and None (a disabled external scan) are not measurement results.
    """

    saved, visited = {}, set()
    selected_target = options.pop("target", None)

    def visit(value):
        # A structural result interface permits standalone callers and notebook reloads.
        if all(hasattr(value, name) for name in ("traces", "targets", "run_id", "metadata", "fits")):
            if not value.targets:
                raise ValueError("The result has no measured targets to export")
            if value.run_id in visited:
                return
            visited.add(value.run_id)
            for target in ((selected_target,) if selected_target is not None else value.targets):
                saved[f"{value.run_id}/{target}"] = save_labber(value, target=target, **options)
            children = value.metadata.get("child_runs", [])
            if children and load is None:
                raise ValueError("Provide load to export scan child runs")
            for run_id in children:
                visit(load(run_id))
        elif isinstance(value, dict):
            if "run_id" in value and "result" not in value:
                if load is None:
                    raise ValueError("Provide load to export run-ID table rows")
                visit(load(value["run_id"]))
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif value is not None and not isinstance(value, (str, int, float, complex, np.generic)):
            raise TypeError(f"Unsupported Labber result type: {type(value).__module__}.{type(value).__qualname__}")

    visit(results)
    if not saved and results is not None and not isinstance(results, (list, tuple, dict)):
        raise TypeError("Expected a V2 measurement result or a collection of results")
    return saved


def save_labber(result, path=None, *, directory=None, target=None, data="iq", comment="", tags=(), save_plot=True, catalog=True, catalog_root=None):
    """Save one target as a Labber log, plus all targets/shots/fits in /metagroup.

    Labber's fastest step is the innermost sweep (or shot index for shots). All dimensions, including
    readout events, remain explicit. Categorical axes use Labber combo labels.
    Select data='shots' to plot individual shots in Labber instead of mean IQ.
    Existing files are never overwritten. Export does not change result.path.
    """
    if target is None:
        if len(result.targets) != 1:
            raise ValueError("Select target explicitly when exporting a multi-target result")
        target = result.targets[0]
    trace = result[target]
    if data not in {"iq", "shots"}:
        raise ValueError("data must be 'iq' or 'shots'")
    values = trace.iq if data == "iq" else trace.shots
    dims = trace.dims if data == "iq" else trace.shot_dims
    if values is None:
        raise ValueError("This result has no captured shots")
    if not dims or any(size == 0 for size in values.shape):
        raise ValueError("Labber export requires nonempty named dimensions")
    for name in (*dims, target):
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("Labber channel names must not contain path separators")
    channel = f"{target}_{data}"
    if channel in dims:
        raise ValueError("Signal channel name conflicts with a sweep dimension")
    sweep_dims = [dim for dim in dims if dim not in {"readout", "state", "shot"}]
    inner = "shot" if data == "shots" and "shot" in dims else (sweep_dims[-1] if sweep_dims else dims[-1])
    export_dims = tuple(dim for dim in dims if dim != inner) + (inner,)
    values = values.transpose([dims.index(dim) for dim in export_dims])
    dims = export_dims
    steps = []
    for dim in reversed(dims):
        axis = np.arange(values.shape[dims.index(dim)]) if dim == "shot" else trace.coords[dim]
        step = {"name": dim, "unit": trace.units.get(dim, ""), "values": axis}
        if axis.dtype.kind in "OUS":
            step["combo_defs"] = [v.decode() if isinstance(v, bytes) else str(v) for v in axis]
            step["values"] = np.arange(len(axis), dtype=float)
        elif axis.dtype.kind not in "biuf" or not np.isfinite(axis).all():
            raise ValueError(f"Labber step {dim} requires finite real coordinates")
        steps.append(step)
    if path is not None and directory is not None:
        raise ValueError("Specify either path or directory, not both")
    if directory is not None:
        date = datetime.fromisoformat(result.created_at.replace("Z", "+00:00")).astimezone()
        folder = Path(directory).expanduser().resolve() / date.strftime("%Y/%m/Data_%m%d")
        path = folder / f"{result.experiment}_{target}_{data}_{result.run_id}.hdf5"
        if Path(path.name).name != path.name or any(c in result.experiment + result.run_id for c in '/\\'):
            raise ValueError("Experiment and run ID must not contain path separators")
    if path is None:
        if result.path is None:
            raise ValueError("Provide an export path for an unsaved result")
        path = result.path.parent / f"{result.experiment}_{target}_{data}.hdf5"
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".hdf5":
        raise ValueError("Labber export path must end in .hdf5")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build privately, then publish atomically without replacing any existing file.
    # Create the published file directly in its destination directory so it
    # inherits that directory's ACL. Windows temporary directories restrict
    # access to their creator; moving/linking their children retains that ACL.
    output = path.parent / f".labber-{uuid4().hex}.hdf5"
    try:
        create_file(str(output), [{"name": channel, "unit": "ADC", "complex": True, "vector": False}], steps)
        with h5py.File(output, "r+") as h5:
            h5.attrs["log_name"] = path.stem
            h5.attrs["comment"] = comment + "\n" + dumps({
                "run_id": result.run_id, "experiment": result.experiment,
                "target": target, "metadata": result.metadata,
                "analysis_status": result.analysis_status, "fits": result.fits,
            })
            supplied_tags = [tags] if isinstance(tags, str) else list(tags)
            h5["Tags"].attrs["Tags"] = np.asarray([target, result.experiment, *supplied_tags], dtype=h5py.string_dtype())
            _write_data(h5, steps, values, channel)
        with h5py.File(output, "r+") as h5:
            group = h5.create_group("metagroup")
            write_record(group, result_record(result), result.traces)
            h5.attrs["export_target"] = target
            h5.attrs["export_kind"] = data
            if save_plot:
                image = BytesIO()
                try:
                    result.plot().savefig(image, format="png", dpi=150)
                except Exception as error:
                    # A plotting failure must not prevent acquired data export.
                    group.attrs["plot_error"] = str(error)
                else:
                    group.create_dataset("plots/analysis.png", data=np.frombuffer(image.getvalue(), dtype=np.uint8), compression="gzip")
        if os.name == "nt":
            os.rename(output, path)  # Windows rename refuses to replace a file.
        else:
            os.link(output, path)
    finally:
        output.unlink(missing_ok=True)
    if catalog:
        register_file_safely(path, catalog_root or directory or path.parent)
    return path


def _write_data(h5, steps, values, channel):
    """Write exact sweep coordinates; never replace rounded/nonuniform axes by linspace."""
    shape = [len(step["values"]) for step in steps]
    rows = values.reshape(-1, shape[0])
    group = h5.create_group("Data")
    group.attrs.update({"Completed": True, "Step dimensions": shape,
                        "Step index": np.arange(len(shape)), "Entries, last trace": shape[0],
                        "Fixed step index": np.array([], dtype=int), "Fixed step values": np.array([])})
    names = [(step["name"], "") for step in steps] + [(channel, "Real"), (channel, "Imaginary")]
    dtype = np.dtype([("name", h5py.string_dtype()), ("info", h5py.string_dtype())])
    group.create_dataset("Channel names", data=np.asarray(names, dtype=dtype))
    dataset = group.create_dataset("Data", shape=(shape[0], len(names), len(rows)), dtype=float, compression="gzip")
    for index, step in enumerate(steps):
        coordinate = np.asarray(step["values"], dtype=float)
        # Step config uses single items so nonuniform values remain inspectable.
        config = h5[f"Step config/{step['name']}"]
        item_dtype = config["Step items"].dtype
        del config["Step items"]
        config.create_dataset("Step items", data=np.array([
            (0, 1, value, value, value, 0, 0, 0, 1, 0, 0) for value in coordinate
        ], dtype=item_dtype))
        if "combo_defs" in step:
            inst = h5["Instrument config/Generic - GPIB: , Step channels at localhost"]
            inst.attrs[f"___{step['name']}___combo_defs"] = np.asarray(step["combo_defs"], dtype=h5py.string_dtype())
        for column in range(len(rows)):
            outer = np.unravel_index(column, tuple(shape[1:]), order="F") if len(shape) > 1 else ()
            dataset[:, index, column] = coordinate if index == 0 else coordinate[outer[index - 1]]
    dataset[:, -2, :] = rows.real.T
    dataset[:, -1, :] = rows.imag.T
    # Export is not a timed acquisition; do not invent per-point timing.
    group.create_dataset("Time stamp", data=np.zeros(len(rows)))
