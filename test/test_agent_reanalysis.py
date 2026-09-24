"""Offline reanalysis must preserve acquisition data and never need a session."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from QickworkspaceV2_agent.analysis.qubit import T1Analysis
from QickworkspaceV2_agent.core.base_analysis import BaseAnalysis
from QickworkspaceV2_agent.core.base_experiment import BaseExperiment
from QickworkspaceV2_agent.core.experiment_data import ExperimentData, QualityFlag
from QickworkspaceV2_agent.nodes.reanalysis import analyze_result


class ReanalysisTests(unittest.TestCase):
    def test_workspace_reanalysis_is_a_new_journaled_run_without_session(self):
        from QickworkspaceV2_agent import Workspace, HardwareSession, RunRequest
        cfg = [{"name": "Q1", "pi_gain_ge": .03, "res_gain_ge": .02}]
        with TemporaryDirectory() as root:
            workspace = Workspace(cfg, data_path=root, session=HardwareSession(object(), {}, root))
            source_data = self.make_t1()
            with patch("QickworkspaceV2_agent.nodes.adapters.execute_experiment", return_value=(None, {"result": source_data})):
                source = workspace.run(RunRequest("t1_ge", "Q1", run={"py_avg": 1}))
            workspace.session = None
            with patch.object(HardwareSession, "activate", side_effect=AssertionError("no hardware session")):
                fresh = workspace.reanalyze(source.run_id)
            self.assertEqual(fresh.status, "completed", fresh.error)
            self.assertEqual(fresh.parent_id, source.run_id)
            self.assertEqual(fresh.data.parent_id, source_data.experiment_id)
            self.assertAlmostEqual(fresh.metrics["T1_us"], 23, places=5)
            self.assertNotEqual(fresh.run_id, source.run_id)
            self.assertEqual(len(workspace.history.list()), 2)
            self.assertEqual(source_data.get_param("T1_us"), -1)
            loaded = Workspace(cfg, data_path=root).get_result(fresh.run_id)
            self.assertEqual(loaded.data.parent_id, source_data.experiment_id)

    def make_t1(self):
        time = np.linspace(0, 80, 81)
        iq = .2 + 1.4 * np.exp(-time / 23.0) + 0j
        return ExperimentData(
            experiment_type="s008_T1_ge", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            x_axis=time, raw_iq=iq,
            raw_data={"readouts": np.stack((iq, iq * .5))},
            axes={"x": {"values": time, "unit": "us"}, "readout": {"values": ["primary", "diagnostic"]}},
            dataset_dims={"iq": ["x"], "readouts": ["readout", "x"]},
            metadata={"node_id": "t1_ge", "analysis_context": {"fit_channel": "real"}, "note": {"keep": [1, 2]}},
            fit_result={"T1_us": (-1, None), "old_fit": (4, None)},
            analysis_data={"old_fit_curve": {"values": np.ones(81), "dims": ["x"]}},
            scalar_result=-1, quality=QualityFlag.BAD, quality_message="old failed fit",
        )

    def test_t1_saved_data_reanalysis_without_session_or_figures(self):
        source = self.make_t1()
        with TemporaryDirectory() as folder:
            path = source.save(Path(folder) / "t1.h5", catalog=False)
            loaded = ExperimentData.load(path)
            original = deepcopy(loaded)
            with patch.object(BaseExperiment, "setup", side_effect=AssertionError("analysis initialized hardware")), patch.object(BaseExperiment, "__init__", side_effect=AssertionError("analysis constructed experiment")), patch.object(plt, "subplots", side_effect=AssertionError("analysis created a figure")):
                result = analyze_result(loaded)
            self.assertAlmostEqual(result.get_param("T1_us"), 23, places=5)
            self.assertEqual(result.quality, QualityFlag.GOOD)
            self.assertNotEqual(result.experiment_id, loaded.experiment_id)
            self.assertGreater(result.timestamp, loaded.timestamp)
            self.assertEqual(result.parent_id, loaded.experiment_id)
            self.assertEqual(result.figures, [])
            self.assertNotIn("old_fit", result.fit_result)
            self.assertNotIn("old_fit_curve", result.analysis_data)
            self.assertEqual(result.quality_message, "")
            self.assertEqual(loaded.fit_result, original.fit_result)
            self.assertEqual(loaded.quality, QualityFlag.BAD)
            np.testing.assert_equal(loaded.raw_iq, original.raw_iq)
            np.testing.assert_equal(result.raw_data["readouts"], loaded.raw_data["readouts"])
            self.assertEqual(result.dataset_dims, loaded.dataset_dims)
            self.assertFalse(np.shares_memory(result.raw_iq, loaded.raw_iq))
            result.raw_data["readouts"][0, 0] = 999j
            result.metadata["note"]["keep"].append(3)
            np.testing.assert_equal(loaded.raw_data["readouts"], original.raw_data["readouts"])
            self.assertEqual(loaded.metadata["note"], {"keep": [1, 2]})

    def test_explicit_local_analysis_handles_data_without_node_metadata(self):
        source = self.make_t1()
        source.metadata.pop("node_id")
        result = analyze_result(source, analysis=T1Analysis())
        self.assertAlmostEqual(result.get_param("T1_us"), 23, places=5)
        with self.assertRaisesRegex(TypeError, "local BaseAnalysis instance"):
            analyze_result(source, analysis="some.module.Analysis")

    def test_missing_and_unsupported_routes_are_actionable(self):
        source = self.make_t1()
        source.metadata.pop("node_id")
        with self.assertRaisesRegex(ValueError, "pass a local analysis instance"):
            analyze_result(source)
        source.metadata["node_id"] = "tomography"
        with self.assertRaisesRegex(ValueError, "dedicated analysis adapter"):
            analyze_result(source)
        self.assertEqual(source.fit_result["old_fit"], (4, None))

    def test_explicit_analysis_failure_does_not_mutate_source(self):
        class FailingAnalysis(BaseAnalysis):
            def _run(self, data):
                data.raw_iq[0] = 123j
                data.metadata["note"]["keep"].append(999)
                raise RuntimeError("new fitting failed")
        source = self.make_t1()
        with self.assertRaisesRegex(RuntimeError, "new fitting failed"):
            analyze_result(source, analysis=FailingAnalysis())
        self.assertAlmostEqual(source.raw_iq[0], 1.6 + 0j)
        self.assertEqual(source.metadata["note"]["keep"], [1, 2])

    def test_single_shot_reanalysis_uses_named_state_axis_and_no_constructor(self):
        from QickworkspaceV2_agent.experiments.setup.single_shot import SingleShot_gef
        rng = np.random.default_rng(12)
        g = rng.normal(-3, .25, 100) + 1j * rng.normal(0, .25, 100)
        e = rng.normal(3, .25, 100) + 1j * rng.normal(0, .25, 100)
        # Stored axis order and state order intentionally differ from the
        # canonical histogram layout, ensuring reconstruction uses labels.
        source = ExperimentData(
            raw_iq=np.column_stack((e, g)), metadata={"node_id": "single_shot", "states": ["e", "g"]},
            axes={"state": {"values": ["e", "g"]}, "shot": {"values": np.arange(100)}},
            dataset_dims={"iq": ["shot", "state"]}, fit_result={"fidelity": (.1, None)},
        )
        with patch.dict("os.environ", {"LOKY_MAX_CPU_COUNT": "1"}), patch.object(SingleShot_gef, "__init__", side_effect=AssertionError("analysis initialized a session")), patch.object(plt, "subplots", side_effect=AssertionError("analysis created a figure")):
            result = analyze_result(source)
        self.assertGreater(result.get_param("fidelity"), .95)
        self.assertEqual(source.get_param("fidelity"), .1)
        self.assertEqual(result.parent_id, source.experiment_id)
        np.testing.assert_equal(result.raw_iq, source.raw_iq)
        self.assertEqual(result.dataset_dims["iq"], ["shot", "state"])
        self.assertEqual(result.metadata["states"], ["e", "g"])
        self.assertEqual(result.axes["prepared_state"]["values"], ["g", "e"])
        self.assertEqual(result.figures, [])


if __name__ == "__main__":
    unittest.main()
