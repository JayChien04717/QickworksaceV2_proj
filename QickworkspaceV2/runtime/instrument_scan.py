"""Durable external instrument scans using the same native measurement runner."""

from dataclasses import dataclass
from uuid import uuid4
import numpy as np

from QickworkspaceV2.backends import AcquisitionCancelled
from QickworkspaceV2.data.models import ExperimentData, TraceData
from QickworkspaceV2.data.store import atomic_json


@dataclass
class InstrumentAxis:
    """The setter owns instrument-specific ramping/settling; read returns actual setpoint.

    No instrument is connected or enabled by construction. Bind an existing driver.
    Limits are in the stated physical unit, and are checked before any write.
    """

    name: str
    unit: str
    read: object
    write: object
    minimum: float
    maximum: float
    resource_id: str

    def validate(self, values):
        if not self.name or not self.resource_id or not callable(self.read) or not callable(self.write):
            raise ValueError("Instrument axis needs a name, resource ID and read/write callables")
        if not np.isfinite([self.minimum, self.maximum]).all() or self.minimum >= self.maximum:
            raise ValueError("Instrument limits must be finite and ordered")
        if not np.isfinite(values).all() or np.any(values < self.minimum) or np.any(values > self.maximum):
            raise ValueError(f"{self.name}: setpoint exceeds declared instrument limits")


def scan_instrument(
    session,
    experiment,
    axis,
    values,
    *,
    target="Q1",
    cancel=None,
    on_progress=None,
    restore=True,
    run_options=None,
    **params,
):
    """Scan external bias, pump frequency or power; save each point and restore on exit.

    The parent journal exists before any instrument write. Completed children and
    partial.h5 survive acquisition errors, cancellation and restoration errors.
    """
    from .session import _hardware_lease

    requested = np.asarray(values, float)
    if requested.ndim != 1 or not len(requested):
        raise ValueError("Instrument setpoints must be a nonempty one-dimensional array")
    axis.validate(requested)
    spec, parsed, _, _, _, _ = session._prepare(experiment, {**params, "target": target}, run_options)
    identifier = uuid4().hex
    metadata = {
        "instrument_scan": axis.name,
        "instrument_resource": axis.resource_id,
        "unit": axis.unit,
        "requested_setpoints": requested.tolist(),
        "actual_setpoints": [],
        "child_runs": [],
        "restore_requested": restore,
        "restored": False,
        "parameters": parsed.model_dump(),
    }
    directory = session.store.directory(identifier)
    collected = []
    original = None
    result = None
    failure = None

    def journal():
        atomic_json(directory / "instrument.json", metadata)

    # Protect the instrument as well as QICK, including time between measurements.
    board_lease = _hardware_lease(session.backend.resource_id)
    with session.backend.lock, board_lease, _hardware_lease("instrument:" + axis.resource_id):
        session.store.begin(identifier, spec.id, metadata)
        try:
            original = float(axis.read())
            axis.validate(np.array([original]))
            metadata["original_setpoint"] = original
            journal()
            for index, value in enumerate(requested):
                if cancel and cancel.is_set():
                    raise AcquisitionCancelled("External scan cancelled between setpoints")
                axis.write(float(value))
                actual = float(axis.read())
                axis.validate(np.array([actual]))
                metadata["last_requested"] = float(value)
                metadata["last_readback"] = actual
                journal()
                child = session._run(
                    spec,
                    parsed.model_dump(),
                    run_options or {},
                    on_progress,
                    cancel,
                    context={
                        "instrument_scan_id": identifier,
                        "instrument": axis.resource_id,
                        "setpoint": actual,
                        "unit": axis.unit,
                    },
                )
                collected.append(child)
                metadata["actual_setpoints"].append(actual)
                metadata["child_runs"].append(child.run_id)
                journal()
                traces = {}
                for q, first in collected[0].traces.items():
                    if axis.name in first.dims:
                        raise ValueError("External instrument axis conflicts with inner sweep dimension")
                    if any(
                        r[q].dims != first.dims
                        or any(not np.array_equal(r[q].coords[d], first.coords[d]) for d in first.dims)
                        for r in collected
                    ):
                        raise ValueError(
                            "Inner axes vary across scan points; children remain available individually"
                        )
                    shots = None if first.shots is None else np.stack([r[q].shots for r in collected])
                    traces[q] = TraceData(
                        np.stack([r[q].iq for r in collected]),
                        (axis.name, *first.dims),
                        {axis.name: np.array(metadata["actual_setpoints"]), **first.coords},
                        {axis.name: axis.unit, **first.units},
                        shots=shots,
                        shot_dims=(axis.name, *first.shot_dims) if shots is not None else (),
                        metadata=first.metadata,
                    )
                result = ExperimentData(
                    spec.id, traces, {**collected[0].metadata, **metadata}, run_id=identifier
                )
                result.save(directory / "partial.h5")
                if on_progress:
                    on_progress(
                        {
                            "state": "scanning",
                            "completed": index + 1,
                            "total": len(requested),
                            "result": result,
                        }
                    )
        except BaseException as exc:
            failure = exc
            metadata["error"] = str(exc)
        finally:
            if restore and original is not None:
                try:
                    axis.write(original)
                    metadata["restored_setpoint"] = float(axis.read())
                    metadata["restored"] = bool(
                        np.isclose(metadata["restored_setpoint"], original, rtol=1e-6, atol=1e-12)
                    )
                    if not metadata["restored"]:
                        raise RuntimeError(
                            "Instrument readback differs from original setpoint after restoration"
                        )
                except BaseException as exc:
                    metadata["restore_error"] = str(exc)
                    failure = failure or exc
            journal()
    if failure:
        session.store.status(
            identifier,
            "cancelled" if isinstance(failure, (AcquisitionCancelled, KeyboardInterrupt)) else "failed",
            str(failure),
        )
        raise failure
    result.metadata.update(metadata)
    session.store.save_acquisition(result)
    session.store.save_analysis(result)
    return result
