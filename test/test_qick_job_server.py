import threading
import time
import unittest

import numpy as np

from QickworkspaceV2.core.base_program import BaseProgram
from JobServer.client import QickJobClient, RemoteJobFailed
from JobServer.adapters.qickworkspacev2 import install_job_adapter, uninstall_job_adapter
from JobServer.remote.qick_job_server import (
    JobCancelled,
    JobValidationError,
    ScheduledQickService,
)


class _FakeSoc:
    def __init__(self):
        self.get_cfg_calls = 0

    def get_cfg(self):
        self.get_cfg_calls += 1
        return {
            "board": "fake",
            "tprocs": [{"type": "qick_processor", "revision": 1}],
            "gens": [],
            "readouts": [],
        }


class QickJobServerTests(unittest.TestCase):
    def setUp(self):
        self.service = ScheduledQickService(soc=_FakeSoc(), job_preflight=False)
        self.client = QickJobClient(self.service)

    def tearDown(self):
        self.service._close_for_test()

    def _wait_status(self, job_id, expected, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.service.status(job_id)
            if status["status"] == expected:
                return status
            time.sleep(0.005)
        self.fail(f"job {job_id} did not reach {expected}: {self.service.status(job_id)}")

    def test_smoke_job_round_trip_does_not_touch_hardware(self):
        submitted = self.client.submit_smoke({"message": "hello"})
        self._wait_status(submitted["job_id"], "SUCCEEDED")
        result = self.client.result(submitted["job_id"])

        self.assertEqual(result["echo"], {"message": "hello"})
        self.assertFalse(result["hardware_touched"])

    def test_backend_discovery_uses_startup_cfg_snapshot(self):
        self.service._close_for_test()
        fake_soc = _FakeSoc()
        self.service = ScheduledQickService(soc=fake_soc, job_preflight=False)
        self.client = QickJobClient(self.service)

        first = self.service.get_cfg()
        first["board"] = "mutated-locally"
        second = self.service.get_cfg()
        self.service.backend_info()

        self.assertEqual(fake_soc.get_cfg_calls, 1)
        self.assertEqual(second["board"], "fake")
        self.assertEqual(
            self.service.backend_info()["execution_model"],
            "single_worker_non_interleaving",
        )

    def test_diagnostic_wait_exposes_running_and_queued_positions(self):
        first = self.client.submit_diagnostic_wait(0.1, owner="user-a")
        self._wait_status(first["job_id"], "RUNNING")
        second = self.client.submit_diagnostic_wait(0.1, owner="user-b")

        first_status = self.client.status(first["job_id"])
        second_status = self.client.status(second["job_id"])
        self.assertEqual(first_status["status"], "RUNNING")
        self.assertEqual(second_status["status"], "QUEUED")
        self.assertEqual(second_status["queue_position"], 1)
        self.assertEqual(second_status["owner"], "user-b")

        self._wait_status(first["job_id"], "SUCCEEDED")
        self._wait_status(second["job_id"], "SUCCEEDED")
        result = self.client.result(second["job_id"])
        self.assertFalse(result["hardware_touched"])
        self.assertEqual(result["waited_seconds"], 0.1)

        recent_ids = {job["job_id"] for job in self.client.list_jobs()}
        self.assertIn(first["job_id"], recent_ids)
        self.assertIn(second["job_id"], recent_ids)

    def test_idempotency_key_reuses_original_job(self):
        first = self.client.submit_smoke("first", idempotency_key="same-request")
        second = self.client.submit_smoke("second", idempotency_key="same-request")

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["reused"])

    def test_worker_executes_one_job_at_a_time_in_fifo_order(self):
        self.service._close_for_test()
        release_first = threading.Event()
        started = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def runner(_service, request):
            nonlocal active, max_active
            label = request["echo"]
            with lock:
                started.append(label)
                active += 1
                max_active = max(max_active, active)
            if label == "first":
                release_first.wait(timeout=1)
            with lock:
                active -= 1
            return label

        self.service = ScheduledQickService(
            soc=_FakeSoc(), job_runner=runner, job_preflight=False
        )
        self.client = QickJobClient(self.service)
        first = self.client.submit_smoke("first")
        self._wait_status(first["job_id"], "RUNNING")
        second = self.client.submit_smoke("second")
        self.assertEqual(self.service.status(second["job_id"])["status"], "QUEUED")

        release_first.set()
        self._wait_status(first["job_id"], "SUCCEEDED")
        self._wait_status(second["job_id"], "SUCCEEDED")
        self.assertEqual(started, ["first", "second"])
        self.assertEqual(max_active, 1)

    def test_queued_job_can_be_cancelled(self):
        self.service._close_for_test()
        release = threading.Event()

        def runner(_service, request):
            if request["echo"] == "blocking":
                release.wait(timeout=1)
            return request["echo"]

        self.service = ScheduledQickService(
            soc=_FakeSoc(), job_runner=runner, job_preflight=False
        )
        self.client = QickJobClient(self.service)
        blocking = self.client.submit_smoke("blocking")
        self._wait_status(blocking["job_id"], "RUNNING")
        queued = self.client.submit_smoke("cancel me")

        response = self.client.cancel(queued["job_id"])
        self.assertEqual(response["status"], "CANCELLED")
        release.set()
        self._wait_status(blocking["job_id"], "SUCCEEDED")
        self.assertEqual(self.service.status(queued["job_id"])["status"], "CANCELLED")

    def test_validation_rejects_unknown_fields_and_excess_rounds(self):
        with self.assertRaises(JobValidationError):
            self.service.submit_job(
                {"schema_version": 1, "job_type": "smoke", "unknown": True}
            )

        request = {
            "schema_version": 1,
            "job_type": "qick_program",
            "program": {
                "format": "qick_dump_prog_json_v1",
                "family": "tprocv2_acquire",
                "payload": "{}",
            },
            "acquisition": {"rounds": 100_001},
        }
        with self.assertRaises(JobValidationError):
            self.service.submit_job(request)

        streaming_accumulated = {
            "schema_version": 1,
            "job_type": "qick_program",
            "program": {
                "format": "qick_dump_prog_json_v1",
                "family": "tprocv2_acquire",
                "payload": "{}",
            },
            "acquisition": {
                "mode": "accumulated",
                "rounds": 2,
                "stream_updates": True,
                "update_every": 1,
            },
        }
        with self.assertRaisesRegex(JobValidationError, "only for decimated"):
            self.service.submit_job(streaming_accumulated)

    def test_streaming_decimated_chunks_and_averages_inside_one_worker_call(self):
        class _FakeProgram:
            def __init__(self):
                self.rounds = []

            def acquire_decimated(self, _soc, **kwargs):
                self.rounds.append(kwargs["rounds"])
                value = float(len(self.rounds))
                return [np.full((3, 2), value)]

        program = _FakeProgram()
        updates = []
        result = self.service._execute_streaming_decimated(
            program,
            {
                "rounds": 5,
                "update_every": 2,
                "stream_updates": True,
                "progress": False,
                "load_envelopes": True,
                "start_src": "internal",
                "remove_offset": True,
            },
            lambda completed, total, partial, phase: updates.append(
                (
                    completed,
                    total,
                    None if partial is None else np.asarray(partial[0]).copy(),
                    phase,
                )
            ),
            lambda: False,
        )

        self.assertEqual(program.rounds, [2, 2, 1])
        completed_updates = [item for item in updates if item[2] is not None]
        last_update_by_completed = {
            item[0]: item for item in completed_updates
        }
        self.assertEqual(
            [item[:2] for item in last_update_by_completed.values()],
            [(2, 5), (4, 5), (5, 5)],
        )
        # Weighted chunk average: (1*2 + 2*2 + 3*1) / 5 = 1.8.
        np.testing.assert_allclose(result[0], 1.8)
        np.testing.assert_allclose(last_update_by_completed[5][2], 1.8)

    def test_streaming_cancel_stops_before_the_next_chunk(self):
        class _FakeProgram:
            def __init__(self):
                self.calls = 0

            def acquire_decimated(self, _soc, **_kwargs):
                self.calls += 1
                return [np.ones((2, 2))]

        program = _FakeProgram()
        with self.assertRaises(JobCancelled):
            self.service._execute_streaming_decimated(
                program,
                {
                    "rounds": 10,
                    "update_every": 2,
                    "stream_updates": True,
                    "progress": False,
                },
                lambda *_args: None,
                lambda: program.calls >= 1,
            )
        self.assertEqual(program.calls, 1)

    def test_failed_job_is_reported_to_client(self):
        self.service._close_for_test()

        def runner(_service, _request):
            raise ValueError("simulated failure")

        self.service = ScheduledQickService(
            soc=_FakeSoc(), job_runner=runner, job_preflight=False
        )
        self.client = QickJobClient(self.service)
        submitted = self.client.submit_smoke("fail")

        with self.assertRaisesRegex(RemoteJobFailed, "simulated failure"):
            self.client.wait(submitted["job_id"], timeout=1, poll_interval=0.005)

    def test_base_program_routes_complete_acquisition_through_job_client(self):
        self.service._close_for_test()
        captured = {}

        def runner(_service, request):
            captured.update(request)
            return [[[[1.0, 2.0]]]]

        self.service = ScheduledQickService(
            soc=_FakeSoc(), job_runner=runner, job_preflight=False
        )
        self.client = QickJobClient(self.service, wait_timeout=1)
        class _Program(BaseProgram):
            def _initialize(self, cfg):
                pass

            def _body(self, cfg):
                pass

        program = _Program.__new__(_Program)
        program.dump_prog = lambda: {"envelopes": [], "example": True}

        install_job_adapter()
        try:
            acquired = program.acquire(
                self.client,
                rounds=3,
                progress=True,
            )
        finally:
            uninstall_job_adapter()

        self.assertEqual(acquired, [[[[1.0, 2.0]]]])
        self.assertEqual(captured["acquisition"]["rounds"], 3)
        self.assertEqual(captured["acquisition"]["mode"], "accumulated")
        self.assertTrue(self.service.status(next(iter(self.service._jobs)))["acknowledged"])


if __name__ == "__main__":
    unittest.main()
