from threading import Event
import json
import pytest
from QickworkspaceV2 import InstrumentAxis, ExperimentData
from QickworkspaceV2.backends import AcquisitionCancelled


def axis_fixture():
    state = {"value": 0.05, "writes": []}

    def write(value):
        state["writes"].append(value)
        state["value"] = value

    return InstrumentAxis("bias", "A", lambda: state["value"], write, -0.2, 0.2, "test-bias"), state


def test_instrument_scan_saves_readback_and_restores(session):
    axis, state = axis_fixture()
    r = session.scan_instrument("resonator_spec", axis, [0.0, 0.1], points=11)
    assert r["Q1"].dims == ("bias", "frequency", "readout")
    assert state["value"] == 0.05
    assert r.metadata["restored"]
    assert len(r.metadata["child_runs"]) == 2
    child = session.load(r.metadata["child_runs"][1])
    assert child.metadata["execution_context"]["setpoint"] == 0.1


def test_instrument_scan_rejects_limits_before_any_write(session):
    axis, state = axis_fixture()
    with pytest.raises(ValueError, match="limits"):
        session.scan_instrument("t1_ge", axis, [1.0])
    assert state["writes"] == []


def test_instrument_scan_cancel_preserves_partial_and_restores(session):
    axis, state = axis_fixture()
    cancel = Event()

    def progress(event):
        if event["state"] == "scanning":
            cancel.set()

    with pytest.raises(AcquisitionCancelled):
        session.scan_instrument("t1_ge", axis, [0.0, 0.1], cancel=cancel, on_progress=progress)
    assert state["value"] == 0.05
    cancelled = next(r for r in session.store.list() if r["status"] == "cancelled")
    directory = session.store.directory(cancelled["id"])
    assert ExperimentData.load(directory / "partial.h5")["Q1"].iq.shape == (1, 81, 1)
    assert json.loads((directory / "instrument.json").read_text())["restored"]


def test_instrument_restore_failure_remains_visible(session):
    axis, state = axis_fixture()
    original_write = axis.write

    def write(value):
        if state["writes"] and value == 0.05:
            raise RuntimeError("restore disconnected")
        original_write(value)

    axis.write = write
    with pytest.raises(RuntimeError, match="restore disconnected"):
        session.scan_instrument("t1_ge", axis, [0.0])
    failed = next(r for r in session.store.list() if r["status"] == "failed")
    assert (session.store.directory(failed["id"]) / "partial.h5").exists()
