"""Protocol tests use numerical/transport fixtures, never physical acquisition."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from QickworkspaceV2 import ExperimentData, TraceData
from QickworkspaceV2.data.transport import to_worker_result
from QickworkspaceV2.runtime.service import create_app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qick_agent.client import WorkerClient


def test_full_artifact_preserves_shots_omitted_from_json(session, tmp_path):
    shots = np.arange(30000).reshape(15000, 2) * (1 + 2j)
    result = ExperimentData('single_shot', {'Q1': TraceData(
        shots.mean(axis=0), ('readout',), {'readout': ['g', 'e']},
        shots=shots, shot_dims=('shot', 'readout'))},
        metadata={'targets': ['Q1'], 'acquisition_status': 'completed', 'test_fixture': True})
    source = session.store.directory(result.run_id) / 'acquisition.h5'
    result.save(source)
    assert 'Q1_shots_I' in to_worker_result(result)['metadata']['omitted_arrays']
    with TestClient(create_app(session)) as http:
        client = WorkerClient('http://testserver', client=http)
        descriptor = client.artifacts(result.run_id)[0]
        destination = client.download_artifact(result.run_id, tmp_path / 'download.h5')
        assert descriptor['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert destination.read_bytes() == source.read_bytes()
        assert http.get('/runs/' + '0' * 32 + '/artifacts').status_code == 404
        assert http.get('/runs/not-a-run/artifacts').status_code == 404
    loaded = ExperimentData.load(destination)
    np.testing.assert_array_equal(loaded['Q1'].shots, shots)
    assert loaded['Q1'].shot_dims == ('shot', 'readout')


def test_stale_submission_does_not_acquire_and_lookup_is_read_only(session, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not acquire')
    monkeypatch.setattr(session.backend, 'acquire', forbidden)
    with TestClient(create_app(session)) as http:
        response = http.post('/experiments/run', json={
            'experiment': 't1_ge', 'request_id': 'stale', 'catalog_revision': 'old'})
        assert response.status_code == 409
        assert response.json()['detail']['code'] == 'catalog_changed'
        assert http.get('/requests/lookup', params={'request_id': 'stale'}).status_code == 404
    assert session.store.list() == []


def test_request_lookup_survives_worker_restart(session):
    with TestClient(create_app(session)) as http:
        client = WorkerClient('http://testserver', client=http)
        job = client.submit('t1_ge', request_id='persisted-request')
        completed = client.wait(job['id'], timeout=15, poll_interval=0.01)
        assert client.by_request_id('persisted-request') == completed
    with TestClient(create_app(session)) as http:
        assert http.get('/requests/lookup', params={'request_id': 'persisted-request'}).json() == completed
        loaded = session.load(completed['run_id'])
        loaded.metadata.pop('acquisition_status', None)
        from unittest.mock import patch
        with patch.object(session, 'load', return_value=loaded):
            wire = http.get(f"/experiments/{completed['id']}/result").json()
        assert wire['acquisition_status'] == 'completed'
        assert wire['status'] == 'success'
        assert 'error' not in wire
        # Same logical request can recover despite an outdated catalog revision.
        repeated = http.post('/experiments/run', json={
            'experiment': 't1_ge', 'request_id': 'persisted-request', 'catalog_revision': 'old'})
        assert repeated.json()['id'] == completed['id']
    assert len(session.store.list()) == 1
