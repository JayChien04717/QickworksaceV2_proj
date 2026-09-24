"""Offline application-boundary tests: adapters are fakes, storage is real."""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from QickworkspaceV2_agent.core.experiment_data import ExperimentData, QualityFlag
from QickworkspaceV2_agent.nodes.contracts import NodeResult, RunRequest, Sweep
from QickworkspaceV2_agent.nodes.session import HardwareSession
from QickworkspaceV2_agent.nodes.validation import SafetyPolicy
from QickworkspaceV2_agent.nodes.workspace import Workspace


class AgentWorkspaceTests(unittest.TestCase):
    def test_seed_change_rejects_previous_proposal_and_readout_change_invalidates_threshold(self):
        result, _ = self.run_fake()
        self.workspace.config.update("qb_freq_ge", 4510.0, q_index="Q1")
        with self.assertRaisesRegex(ValueError, "Seed configuration changed"):
            self.workspace.commit(result)
        res, _ = self.run_fake(self.request("res_spec_ge"), data=self.data(metrics={"f_res[MHz]": (6510., .01)}))
        self.assertIn("threshold", res.suggested_updates)
        self.assertIsNone(res.suggested_updates["threshold"])
        self.workspace.commit(res)
        self.assertIsNone(self.workspace.calibrations("Q1")["values"]["threshold"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.seed = [{"name": "Q1", "reps": 1, "steps": 5, "relax_delay": 1.0,
                      "pi_gain_ge": .03, "res_gain_ge": .02, "pi_gain_ef": .025,
                      "qb_freq_ge": 4500.0, "qb_freq_ef": 4250.0,
                      "res_freq_ge": 6500.0, "cooling": False}]
        self.session = HardwareSession(object(), {}, self.tmp.name)
        self.workspace = Workspace(self.seed, data_path=self.tmp.name, session=self.session)

    def request(self, experiment="power_rabi_ge", **options):
        return RunRequest(experiment, "Q1", **options)

    @staticmethod
    def data(metrics=None, *, quality=QualityFlag.GOOD, interrupted=False):
        return ExperimentData(
            experiment_type="offline_contract_test", raw_iq=np.array([1 + 2j, 3 - 4j, 5 + 6j]),
            x_axis=np.array([.01, .02, .03]),
            fit_result=metrics or {"pi_gain": (.04, .001), "pi2_gain": (.02, .001)},
            quality=quality, interrupted=interrupted,
        )

    def run_fake(self, request=None, *, workspace=None, data=None):
        data = self.data() if data is None else data
        fake = SimpleNamespace(result=data)
        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment", return_value=(fake, {"result": data})) as execute:
            result = (workspace or self.workspace).run(request or self.request())
        return result, execute

    def test_invalid_json_requests_fail_before_adapter_or_journal(self):
        invalid = [
            self.request(run={"py_avg": True}), self.request(run={"py_avg": 0}),
            self.request(run={"py_avg": 1, "unknown": True}),
            self.request(init={"arbitrary_driver": "os.system"}),
            self.request(config={"qb_freq_ge": float("nan")}),
            self.request(config={"opaque": object()}),
            self.request(persist="yes"), self.request(request_id=""),
            self.request(config={"pi_gain_ge": Sweep("gainloop", 0, .08, points=1)}),
            {"experiment": "t1_ge", "qubit": "Q1", "unexpected": 1},
        ]
        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment") as execute:
            for request in invalid:
                with self.subTest(request=repr(request)):
                    with self.assertRaises((ValueError, TypeError)):
                        self.workspace.run(request)
            execute.assert_not_called()
        self.assertEqual(self.workspace.history.list(), [])

    def test_preview_checks_every_gain_endpoint_and_nested_gain(self):
        for value in (.1, -.1, [0, .1], {"$sweep": {"loop": "gainloop", "start": .1, "stop": .01}},
                      {"$sweep": {"loop": "gainloop", "start": .01, "stop": -.1}}):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.workspace.preview(self.request(config={"special_gain": value}))
        with self.assertRaises(ValueError):
            self.workspace.preview(self.request(config={"pulse": {"gain": .2}}))
        with self.assertRaises(ValueError):
            self.workspace.preview(self.request("single_shot_opt", run={"SHOTS": 10, "sweep_para": {"gain": [.01, .1]}}))
        accepted = self.workspace.preview(self.request(config={"pi_gain_ge": Sweep("gainloop", -.09, .09, points=7)}))
        self.assertEqual(accepted["effective_config"]["steps"], 7)

    def test_config_seed_detached_and_accepted_values_overlay_before_request(self):
        original_seed = deepcopy(self.seed)
        source_config = {"pi_gain_ge": .035}
        request = self.request(config=source_config, persist=False)

        def execute(spec, config, run, **kwargs):
            self.assertEqual(config["pi_gain_ge"], .035)
            config["pi_gain_ge"] = .08
            run["py_avg"] = 999
            data = self.data()
            return SimpleNamespace(result=data), {"result": data}

        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment", side_effect=execute):
            result = self.workspace.run(request)
        self.assertEqual(source_config, {"pi_gain_ge": .035})
        self.assertEqual(request.run, {"py_avg": 1})
        self.assertEqual(self.seed, original_seed)
        self.assertEqual(self.workspace.config.get_qubit("Q1")["pi_gain_ge"], .03)
        self.workspace.commit(result)
        self.assertEqual(self.workspace.preview(self.request())["effective_config"]["pi_gain_ge"], .04)
        self.assertEqual(self.workspace.preview(self.request(config={"pi_gain_ge": .045}))["effective_config"]["pi_gain_ge"], .045)
        self.seed[0]["pi_gain_ge"] = .09
        self.assertEqual(self.workspace.config.get_qubit("Q1")["pi_gain_ge"], .03)

    def test_ef_measurement_commits_only_ef_gain(self):
        result, _ = self.run_fake(self.request("power_rabi_ef", persist=False))
        self.assertEqual(result.suggested_updates, {"pi_gain_ef": .04, "pi2_gain_ef": .02})
        self.workspace.commit(result)
        effective = self.workspace.preview(self.request())["effective_config"]
        self.assertEqual(effective["pi_gain_ge"], .03)
        self.assertEqual(effective["pi_gain_ef"], .04)

    def test_hdf5_roundtrip_preserves_complex_raw_data_and_journal_snapshot(self):
        data = self.data()
        result, execute = self.run_fake(data=data)
        self.assertEqual(result.status, "completed", result.error)
        self.assertEqual(execute.call_count, 1)
        path = Path(result.artifacts["data:result"])
        self.assertTrue(path.is_file())
        fresh = Workspace(self.seed, data_path=self.tmp.name)
        loaded = fresh.get_result(result.run_id)
        np.testing.assert_array_equal(loaded.data.raw_iq, data.raw_iq)
        self.assertEqual(loaded.data.config, {})
        self.assertEqual(loaded.data.metadata["run_id"], result.run_id)
        self.assertEqual(loaded.data.session_id, result.run_id)
        record = fresh.history.get(result.run_id)
        self.assertEqual(record["snapshot"]["config"]["pi_gain_ge"], .03)
        self.assertEqual(len(loaded.data.metadata["config_snapshot_sha256"]), 64)
        json.dumps(loaded.to_dict(), allow_nan=False)

    def test_two_workspaces_reject_stale_proposal_and_commit_idempotently(self):
        second = Workspace(self.seed, data_path=self.tmp.name, session=self.session)
        first_result, _ = self.run_fake(self.request(persist=False))
        stale_result, _ = self.run_fake(self.request(persist=False), workspace=second)
        self.assertEqual(self.workspace.commit(first_result), 1)
        self.assertEqual(second.commit(first_result.run_id), 1)
        with self.assertRaisesRegex(ValueError, "Calibration changed"):
            second.commit(stale_result)
        self.assertEqual(second.calibrations("Q1")["revision"], 1)
        newer, _ = self.run_fake(self.request(persist=False), workspace=second)
        self.assertEqual(newer.calibration_revision, 1)
        self.assertEqual(second.commit(newer), 2)

    def test_mutating_returned_proposal_cannot_change_stored_commit(self):
        result, _ = self.run_fake(self.request(persist=False))
        result.suggested_updates["pi_gain_ge"] = .09
        result.suggested_updates["arbitrary_parameter"] = 123
        result.metrics["pi_gain"] = .09
        self.workspace.commit(result)
        self.assertEqual(self.workspace.calibrations("Q1")["values"], {"pi_gain_ge": .04, "pi2_gain_ge": .02})

    def test_warning_interrupted_and_failed_results_cannot_commit(self):
        for quality, interrupted in ((QualityFlag.WARNING, False), (QualityFlag.BAD, False), (QualityFlag.GOOD, True)):
            with self.subTest(quality=quality, interrupted=interrupted):
                result, _ = self.run_fake(self.request(persist=False), data=self.data(quality=quality, interrupted=interrupted))
                self.assertEqual(result.suggested_updates, {})
                with self.assertRaises(ValueError):
                    self.workspace.commit(result)

    def test_request_id_replay_never_reacquires_in_same_or_new_workspace(self):
        request = self.request(request_id="measure-once", persist=False)
        first, execute = self.run_fake(request)
        self.assertEqual(execute.call_count, 1)
        second = Workspace(self.seed, data_path=self.tmp.name, session=self.session)
        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment") as repeated:
            self.assertEqual(self.workspace.run(request).run_id, first.run_id)
            self.assertEqual(second.run(request).run_id, first.run_id)
            with self.assertRaisesRegex(ValueError, "different request"):
                second.run(self.request(request_id="measure-once", run={"py_avg": 2}, persist=False))
            repeated.assert_not_called()

    def test_uncertain_running_journal_row_is_returned_without_reacquisition(self):
        request = self.request(request_id="interrupted-process", persist=False)
        row = NodeResult(request.request_id, request.experiment, "Q1", "running")
        self.workspace.history.start(request.to_dict(), {"config": self.seed[0]}, row.to_dict())
        fresh = Workspace(self.seed, data_path=self.tmp.name, session=self.session)
        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment") as execute:
            replay = fresh.run(request)
            self.assertEqual(replay.status, "running")
            execute.assert_not_called()

    def test_concurrent_duplicate_request_claims_once_and_refreshes_running_result(self):
        from QickworkspaceV2_agent.nodes.history import History

        second = Workspace(self.seed, data_path=self.tmp.name, session=self.session)
        request = self.request(request_id="simultaneous", persist=False)
        barrier = threading.Barrier(2)
        loser_returned = threading.Event()
        original_start = History.start

        def simultaneous_start(history, *args):
            barrier.wait(timeout=5)
            return original_start(history, *args)

        def execute(*args, **kwargs):
            self.assertTrue(loser_returned.wait(timeout=5))
            data = self.data()
            return SimpleNamespace(result=data), {"result": data}

        def call(workspace):
            result = workspace.run(request)
            if result.status == "running":
                loser_returned.set()
            return result

        with patch.object(History, "start", simultaneous_start), patch(
            "QickworkspaceV2_agent.nodes.adapters.execute_experiment", side_effect=execute
        ) as acquisition, ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(call, workspace) for workspace in (self.workspace, second)]
            results = [future.result(timeout=10) for future in futures]
        self.assertEqual(acquisition.call_count, 1)
        self.assertEqual({result.status for result in results}, {"running", "completed"})
        self.assertEqual(second.get_result(request.request_id).status, "completed")
        self.assertEqual(self.workspace.get_result(request.request_id).status, "completed")

    def test_single_shot_phase_threshold_commit_requires_zero_phase_recipe(self):
        for phase, expected in ((0, {"res_phase": 25.0, "threshold": .01}), (10, {})):
            with self.subTest(phase=phase):
                data = self.data({"rotation_deg": (25.0, None), "threshold": (.01, None)})
                request = self.request("single_shot", config={"res_phase": phase}, run={"SHOTS": 10}, persist=False)
                result, _ = self.run_fake(request, data=data)
                self.assertEqual(result.suggested_updates, expected)

    def test_ge_frequency_updates_its_mixer_but_ef_does_not(self):
        for experiment, expected in (("qubit_spec_ge", {"qb_freq_ge": 4499.0, "qb_mixer": 4499.0}),
                                     ("qubit_spec_ef", {"qb_freq_ef": 4499.0})):
            with self.subTest(experiment=experiment):
                result, _ = self.run_fake(self.request(experiment, persist=False), data=self.data({"f0_MHz": (4499.0, .1)}))
                self.assertEqual(result.suggested_updates, expected)

    def test_save_failure_retains_raw_in_memory_and_failure_in_journal(self):
        data = self.data()
        with patch.object(data, "save", side_effect=OSError("disk full")):
            result, _ = self.run_fake(data=data)
        self.assertEqual(result.status, "failed")
        self.assertIn("disk full", result.error)
        self.assertIs(result.data, data)
        np.testing.assert_array_equal(result.data.raw_iq, [1 + 2j, 3 - 4j, 5 + 6j])
        self.assertFalse(any(key.startswith("data:") for key in result.artifacts))
        self.assertEqual(self.workspace.history.get(result.run_id)["result"]["status"], "failed")
        self.assertEqual(result.suggested_updates, {})

    def test_analysis_exception_preserves_attached_acquisition_result(self):
        data = self.data(quality=QualityFlag.BAD)
        error = RuntimeError("cannot fit acquired trace")
        error.experiment_object = SimpleNamespace(result=data)
        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment", side_effect=error):
            result = self.workspace.run(self.request())
        self.assertEqual(result.status, "failed")
        self.assertTrue(Path(result.artifacts["data:primary"]).is_file())
        np.testing.assert_array_equal(result.data.raw_iq, data.raw_iq)

    def test_actual_base_experiment_post_fit_failure_retains_raw_data(self):
        from QickworkspaceV2_agent.core.base_experiment import BaseExperiment
        from QickworkspaceV2_agent.core.experiment_components import AcquisitionResult

        raw = np.array([1 + 2j, 3 + 4j, 5 + 6j])

        class FailedFitExperiment(BaseExperiment):
            EXPT_NAME = "offline_failed_fit"

            def _create_program(self):
                return object()

            def _extract_sweep_axis(self, program):
                return np.arange(3)

            def _acquire(self, program, x_vals, y_vals, options):
                return AcquisitionResult(raw_iq=raw.copy(), quality=QualityFlag.GOOD, avg_count=1)

            def _post_fit(self, x_vals):
                raise RuntimeError("post_fit fails after acquisition")

        with patch("QickworkspaceV2_agent.nodes.adapters._resolve_class", return_value=FailedFitExperiment):
            result = self.workspace.run(self.request("t1_ge"))
        self.assertEqual(result.status, "failed")
        self.assertIn("post_fit fails after acquisition", result.error)
        np.testing.assert_array_equal(result.data.raw_iq, raw)
        self.assertTrue(Path(result.artifacts["data:primary"]).is_file())

    def test_session_bridge_restores_globals_after_execution_exception(self):
        from QickworkspaceV2_agent.core.base_experiment import BaseExperiment
        names = ("_soc", "_soccfg", "_data_path", "_session_name")
        previous = {key: getattr(BaseExperiment, key) for key in names}
        with self.assertRaisesRegex(RuntimeError, "inside context"):
            with self.session.activate():
                self.assertIs(BaseExperiment._soc, self.session.soc)
                raise RuntimeError("inside context")
        for key, value in previous.items():
            self.assertIs(getattr(BaseExperiment, key), value)

    def test_tool_api_disabled_commands_and_enabled_json_result(self):
        from QickworkspaceV2_agent.nodes.tools import ToolAPI

        readonly = ToolAPI(self.workspace)
        names = {item["function"]["name"] for item in readonly.definitions()}
        self.assertNotIn("run_experiment", names)
        self.assertNotIn("commit_calibration", names)
        with self.assertRaisesRegex(ValueError, "disabled"):
            readonly.call("run_experiment", self.request(persist=False).to_dict())
        with self.assertRaises(ValueError):
            readonly.call("list_experiments", {"arbitrary": 1})
        preview = readonly.call("preview_experiment", self.request(persist=False).to_dict())
        self.assertEqual(preview["effective_config"]["pi_gain_ge"], .03)
        enabled = ToolAPI(self.workspace, allow_execution=True, allow_commit=True)
        data = self.data()
        with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment", return_value=(SimpleNamespace(result=data), {"result": data})) as execute:
            result = enabled.call("run_experiment", self.request(persist=False).to_dict())
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("results", result)
        self.assertNotIn("raw_iq", result)
        json.dumps(result, allow_nan=False)
        self.assertEqual(enabled.call("commit_calibration", {"run_id": result["run_id"]})["revision"], 1)
        self.assertEqual(readonly.call("get_result", {"run_id": result["run_id"]})["metrics"]["pi_gain"], .04)

    def test_rest_openapi_validation_and_enabled_execution(self):
        try:
            from fastapi.testclient import TestClient
            from QickworkspaceV2_agent.service.api import create_app
        except ImportError:
            self.skipTest("Optional FastAPI/httpx dependencies are unavailable")

        readonly_app = create_app(self.workspace)
        schema = readonly_app.openapi()
        self.assertIn("/runs/preview", schema["paths"])
        json.dumps(schema, allow_nan=False)
        with TestClient(readonly_app) as client:
            self.assertEqual(client.get("/experiments").status_code, 200)
            self.assertEqual(client.post("/runs", json=self.request(persist=False).to_dict()).status_code, 403)
            self.assertEqual(client.post("/runs/preview", json={**self.request().to_dict(), "extra": True}).status_code, 422)
            self.assertEqual(client.post("/runs/preview", json={**self.request().to_dict(), "persist": "true"}).status_code, 422)
        app = create_app(self.workspace, allow_execution=True, allow_commit=True)
        data = self.data()
        with TestClient(app) as client, patch(
            "QickworkspaceV2_agent.nodes.adapters.execute_experiment", return_value=(SimpleNamespace(result=data), {"result": data})
        ) as execute:
            invalid = client.post("/runs", json=self.request(run={"py_avg": True}, persist=False).to_dict())
            self.assertEqual(invalid.status_code, 422)
            execute.assert_not_called()
            response = client.post("/runs", json=self.request(request_id="http-once", persist=False).to_dict())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "completed")
            self.assertNotIn("results", response.json())
            replay = client.post("/runs", json=self.request(request_id="http-once", persist=False).to_dict())
            self.assertEqual(replay.status_code, 200)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(client.get("/runs/http-once").json()["metrics"]["pi_gain"], .04)
            self.assertEqual(client.post("/runs/http-once/commit").json()["revision"], 1)


class AgentPulsePreservationTests(unittest.TestCase):
    def test_program_initialization_and_pulse_methods_are_ast_identical(self):
        root = Path(__file__).resolve().parents[1]
        original = root / "QickworkspaceV2"
        copied = root / "QickworkspaceV2_agent"
        protected = {"_initialize", "_body", "apply_cool", "cooling_body", "measure", "activate_reset"}
        compared = 0
        for source in [original / "core" / "base_program.py", *sorted((original / "experiments").rglob("*.py")), *sorted((original / "experiments_mux").rglob("*.py"))]:
            target = copied / source.relative_to(original)
            source_tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            target_tree = ast.parse(target.read_text(encoding="utf-8-sig").replace("QickworkspaceV2_agent", "QickworkspaceV2"))

            def methods(tree):
                return {(cls.name, method.name): ast.dump(method, include_attributes=False)
                        for cls in tree.body if isinstance(cls, ast.ClassDef)
                        for method in cls.body if isinstance(method, ast.FunctionDef)
                        and (method.name in protected or method.name.startswith("setup_"))}

            before, after = methods(source_tree), methods(target_tree)
            for key, body in before.items():
                with self.subTest(file=str(source.relative_to(original)), method=key):
                    self.assertEqual(after.get(key), body)
                compared += 1
        self.assertGreater(compared, 100)


if __name__ == "__main__":
    unittest.main()
