import ast
import base64
import json
from threading import Event
import pytest
from fastapi.testclient import TestClient
from QickworkspaceV2.runtime.service import create_app
from QickworkspaceV2.integrations.client import WorkerClient
from QickworkspaceV2.integrations.nvidia import export_wrappers, to_blueprint


def test_blueprint_wrapper_discovery_shape(session, tmp_path):
    paths = export_wrappers(session.registry, tmp_path / "scripts")
    assert len(paths) == len(session.catalog())
    for path in paths:
        tree = ast.parse(path.read_text())
        public = [n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
        assert len(public) == 1
        assert isinstance(public[0].returns, ast.Name) and public[0].returns.id == "dict"
        assert all(a.annotation is not None for a in public[0].args.args)


def test_blueprint_has_real_arrays_and_image(accepted_record):
    payload = to_blueprint(accepted_record)
    json.dumps(payload, allow_nan=False)
    assert payload["status"] == "success"
    assert payload["data"]["Q1_I"]["type"] == "array"
    assert base64.b64decode(payload["plots"][0]["data"]).startswith(b"\x89PNG")
    assert payload["plots"][0]["name"] == "t1_ge"


@pytest.mark.parametrize(
    "dims,shape,coords",
    [
        (("readout",), (2,), {"readout": ["g", "e"]}),
        (("state", "readout"), (2, 1), {"state": [0, 1], "readout": ["measurement"]}),
    ],
)
def test_blueprint_shots_preserve_named_axes_after_roundtrip(dims, shape, coords, tmp_path):
    import numpy as np
    from QickworkspaceV2 import ExperimentData, TraceData

    shots = np.arange(4 * np.prod(shape)).reshape(4, *shape) * (1 + 2j)
    result = ExperimentData(
        "single_shot",
        {"Q1": TraceData(shots.mean(axis=0), dims, coords, shots=shots, shot_dims=("shot", *dims))},
        metadata={"targets": ["Q1"], "iq_process": "abs", "test_fixture": True},
    )
    path = tmp_path / "shots.h5"
    result.save(path)
    loaded = ExperimentData.load(path)
    payload = json.loads(json.dumps(to_blueprint(loaded), allow_nan=False))
    for name, values in (("Q1_shots_I", shots.real), ("Q1_shots_Q", shots.imag)):
        np.testing.assert_array_equal(payload["data"][name]["value"], values)
        assert payload["data"][name]["dims"] == ["shot", *dims]
        assert payload["metadata"]["array_metadata"][name]["shape"] == [4, *shape]
    assert payload["metadata"]["array_metadata"]["Q1_readout"]["labels"] == coords["readout"]
    assert payload["metadata"]["array_metadata"]["Q1_readout"]["dims"] == ["readout"]
    if "state" in dims:
        assert payload["data"]["Q1_state"]["value"] == [0, 1]
        assert payload["data"]["Q1_state"]["dims"] == ["state"]
    np.testing.assert_array_equal(ExperimentData.load(path)["Q1"].shots, shots)


def test_blueprint_reports_omitted_shots_and_honors_export_budget():
    import numpy as np
    from QickworkspaceV2 import ExperimentData, TraceData

    result = ExperimentData(
        "single_shot",
        {"Q1": TraceData(np.ones(2), ("readout",), {"readout": ["g", "e"]},
                         shots=np.ones((3, 2)), shot_dims=("shot", "readout"))},
        metadata={"targets": ["Q1"], "iq_process": "abs", "test_fixture": True},
    )
    payload = to_blueprint(result, max_array_values=6)
    descriptors = payload["metadata"]["array_metadata"]
    exported = sum(np.asarray(item["value"]).size for item in payload["data"].values() if item["type"] == "array")
    exported += sum(np.asarray(item["labels"]).size for item in descriptors.values() if "labels" in item)
    assert exported == 6
    assert descriptors["Q1_readout"]["labels"] == ["g", "e"]
    omitted = payload["metadata"]["omitted_arrays"]
    assert omitted["Q1_readout"]["reason"] == "non_numeric_dtype"
    for name in ("Q1_shots_I", "Q1_shots_Q"):
        assert name not in payload["data"]
        assert omitted[name] == {"dims": ["shot", "readout"], "shape": [3, 2], "unit": "ADC",
                                 "reason": "array_export_limit", "required_values": 6}


def test_worker_idempotency_and_payload(session):
    with TestClient(create_app(session)) as http:
        client = WorkerClient("http://testserver", client=http)
        first = client.submit("t1_ge", parameters={"target": "Q1,Q2"}, request_id="same-request")
        second = client.submit("t1_ge", parameters={"target": "Q1,Q2"}, request_id="same-request")
        assert first["id"] == second["id"]
        conflict = http.post(
            "/experiments/run",
            json={"experiment": "t1_ge", "parameters": {"target": "Q1"}, "request_id": "same-request"},
        )
        assert conflict.status_code == 409
        payload = client.wait(first["id"], timeout=15, poll_interval=0.01)
        assert payload["status"] == "failed"  # The constant IQ fixture must fail fit quality.
        assert len(session.store.list()) == 1
        assert payload["metadata"]["calibration_updated"] is False
        assert payload["metadata"]["acquisition_status"] == "completed"
        assert payload["metadata"]["quality"] == "bad"
        assert payload["metadata"]["resolved_config"]["targets"] == ["Q1", "Q2"]
        assert payload["metadata"]["device_snapshot"]["device"]["device_id"] == session.device.config.device_id
        assert payload["metadata"]["run_options"]["reps"] == session.defaults.reps
        invalid = http.post("/experiments/run", json={"experiment": "t1_ge", "parameters": {"pointz": 100}})
        assert invalid.status_code == 422


def test_worker_check_compiles_requested_run_options(session, monkeypatch):
    compile_program = session.backend.compile
    compiled = []

    def capture(plan):
        compiled.append((plan.cfg["reps"], plan.cfg["relax_delay"]))
        return compile_program(plan)

    monkeypatch.setattr(session.backend, "compile", capture)
    options = {"reps": 7, "soft_avgs": 3, "relax_delay_us": 12.5, "iq_process": "real"}
    with TestClient(create_app(session)) as http:
        response = http.post("/experiments/check", json={"experiment": "t1_ge", "run_options": options})
    assert response.status_code == 200
    assert response.json()["ready"]
    assert response.json()["run_options"] == options
    assert compiled == [(7, 12.5)]
    assert session.store.list() == []


@pytest.mark.parametrize(
    "payload",
    [
        {"parameters": {"reps": 2}},
        {"run_options": {"unknown": 2}},
        {"run_options": {"soft_avgs": 0}},
    ],
)
def test_worker_check_and_run_reject_same_invalid_options(session, payload):
    with TestClient(create_app(session)) as http:
        body = {"experiment": "t1_ge", **payload}
        checked = http.post("/experiments/check", json=body)
        submitted = http.post("/experiments/run", json=body)
    assert not checked.json()["ready"]
    assert submitted.status_code == 422
    assert session.store.list() == []


def test_worker_config_snapshot_is_read_only(session, monkeypatch):
    from pathlib import Path

    session.project_path = Path("lab/project.yaml").resolve()
    original = session.device.snapshot()

    def no_connect(*args, **kwargs):
        raise AssertionError("Configuration inspection must not contact hardware")

    monkeypatch.setattr(session.backend, "acquire", no_connect)
    with TestClient(create_app(session)) as http:
        snapshot = http.get("/config").json()
        health = http.get("/health").json()
    assert snapshot["hardware"] == original["hardware"]
    assert snapshot["device"] == json.loads(json.dumps(original["device"]))
    assert snapshot["defaults"] == session.defaults.model_dump()
    assert snapshot["calibration_revision"] == 0
    assert set(snapshot["connection"]) == {"ns_host", "ns_port", "proxy_name"}
    assert snapshot["capabilities"]["status"] == "available"
    assert snapshot["capabilities"]["hardware_verified"] is False
    assert health["hardware_verified"] is False
    assert session.device.snapshot() == original
    assert session.calibration.snapshot() == (0, {})
    assert session.store.list() == []


def test_worker_progress_is_visible_and_survives_restart(session, monkeypatch):
    from QickworkspaceV2 import BaseProgram

    original_acquire = BaseProgram.acquire
    second_round, release = Event(), Event()
    calls = 0

    def acquire(program, soc, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_round.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release acquisition transport fixture")
        return original_acquire(program, soc, **kwargs)

    monkeypatch.setattr(BaseProgram, "acquire", acquire)
    with TestClient(create_app(session)) as http:
        client = WorkerClient("http://testserver", client=http)
        first = client.submit("t1_ge", run_options={"soft_avgs": 2})
        try:
            assert second_round.wait(10)
            active = client.status(first["id"])
            assert active["status"] == "running"
            assert active["progress"] == {"phase": "acquiring", "completed": 1, "total": 2}
            assert active["run_id"] is not None
        finally:
            release.set()
        client.wait(first["id"], timeout=15, poll_interval=0.01)
        completed = client.status(first["id"])
        assert completed["progress"] == {"phase": "completed", "completed": 2, "total": 2}
    with TestClient(create_app(session)) as restarted:
        restored = restarted.get(f"/experiments/{first['id']}").json()
    assert restored == completed
