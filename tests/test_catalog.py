import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel

from QickworkspaceV2.experiments.base import ExperimentSpec, Parameters
from QickworkspaceV2.runtime.catalog import catalog_payload
from QickworkspaceV2.runtime.service import create_app


def test_worker_catalog_matches_offline_test_fixture_and_does_not_acquire(session):
    original = session.registry.catalog()
    with TestClient(create_app(session)) as client:
        payload = client.get("/catalog").json()
        assert client.get("/experiments").status_code == 404
    assert payload == catalog_payload(session.registry)
    fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/v2_catalog.json"
    assert payload == json.loads(fixture.read_text(encoding="utf-8"))
    assert session.registry.catalog() == original
    assert session.store.list() == []


def test_custom_experiment_schema_is_exposed_without_generating_files(session):
    class Options(BaseModel):
        delay: float

    class CustomParameters(Parameters):
        options: Options

    session.registry.register(ExperimentSpec("custom_experiment", CustomParameters, lambda ctx, p: None))
    with TestClient(create_app(session)) as client:
        entries = client.get("/catalog").json()["experiments"]
    custom = next(e for e in entries if e["id"] == "custom_experiment")
    schema = custom["parameters"]
    assert schema["properties"]["options"]["$ref"] == "#/$defs/Options"
    assert schema["$defs"]["Options"]["properties"]["delay"]["type"] == "number"
    assert {"run_options", "request_id"} <= schema["properties"].keys()
    assert session.store.list() == []
