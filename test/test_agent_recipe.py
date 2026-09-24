"""Calibration recipes are inspectable requests and never connect implicitly."""
import tempfile
import unittest
from unittest.mock import patch

from QickworkspaceV2_agent import Workspace, SafetyPolicy
from QickworkspaceV2_agent.calibration import AutoCalibrate
from QickworkspaceV2_agent.nodes.catalog import get_spec
from QickworkspaceV2_agent.nodes.validation import validate_schema


class RecipeTests(unittest.TestCase):
    def test_optimizer_does_not_approve_a_poor_readout_grid(self):
        import numpy as np
        from QickworkspaceV2_agent.experiments.setup.single_shot import SingleShot_ge_opt
        from QickworkspaceV2_agent.core.experiment_data import QualityFlag
        optimizer = object.__new__(SingleShot_ge_opt)
        optimizer.cfg = {"name": "Q1"}
        optimizer.data = {}
        optimizer.length_sweep = np.array([2.])
        optimizer.gain_sweep = np.array([.03])
        optimizer.shots = 3
        optimizer.fixed_frequency = 6700.
        for name in ("I_g_array", "Q_g_array", "I_e_array", "Q_e_array"):
            setattr(optimizer, name, np.zeros((1, 1, 3)))
        metrics = dict(fid=.5, e_to_g_error=.5, g_to_e_error=.5,
                       e_core_fraction=1., g_core_fraction=1., e_tail_fraction=0.,
                       g_tail_fraction=0., readout_score=.25, threshold=0., rotation_deg=0.)
        with patch("QickworkspaceV2_agent.experiments.setup.single_shot.fast_histogram_metrics", return_value=metrics):
            optimizer.analyze()
        self.assertEqual(optimizer.result.quality, QualityFlag.WARNING)
        self.assertEqual(optimizer.result.raw_iq.shape, (1, 1, 2, 3))

    def test_mux_cooling_checks_each_slot(self):
        SafetyPolicy().validate({"cooling": [False, False], "res_gains": [0.02, 0.03]})
        with self.assertRaises(ValueError):
            SafetyPolicy().validate({"cooling": [False, True]})
        with self.assertRaises(ValueError):
            SafetyPolicy().validate({"cooling": "false"})

    def test_recipe_is_portable_and_never_reuses_old_threshold(self):
        cfg = [{"name": "Q1", "res_freq_ge": 6700., "qb_freq_ge": 4000.,
                "ro_length": 2., "res_gain_ge": 0.02, "threshold": 0.5}]
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(cfg, data_path=root)
            with patch.object(workspace, "run", side_effect=AssertionError("must not acquire")):
                flow = AutoCalibrate(workspace, "Q1").plan()
            self.assertEqual(len(flow.nodes), 8)
            flow.validate()
            for node in flow.nodes:
                self.assertIsNone(node.request.config["threshold"])
                spec = get_spec(node.request.experiment)
                validate_schema(node.request.to_dict()["config"], spec.config_schema)
                validate_schema(node.request.run, spec.run_schema)
            self.assertEqual(flow.nodes[-1].request.config["res_phase"], 0)
            self.assertTrue(all(node.commit for node in flow.nodes))
            with self.assertRaises(ValueError):
                AutoCalibrate(workspace, "Q1").plan(skip=["typo"])


if __name__ == "__main__":
    unittest.main()
