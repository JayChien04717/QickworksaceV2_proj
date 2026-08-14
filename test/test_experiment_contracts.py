import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.core.base_experiment import BaseExperiment
from QickworkspaceV2.core.experiment_data import ExperimentData
from QickworkspaceV2.experiments.characterization.allxy import AllXY
from QickworkspaceV2.experiments.qubit_ge.aae import PowerRabiChevron


class ExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_runtime = (
            BaseExperiment._soc,
            BaseExperiment._soccfg,
            BaseExperiment._data_path,
        )
        BaseExperiment.setup(object(), object(), "/tmp")

    @classmethod
    def tearDownClass(cls):
        (
            BaseExperiment._soc,
            BaseExperiment._soccfg,
            BaseExperiment._data_path,
        ) = cls.previous_runtime

    def test_package_never_imports_legacy_qick_workspace(self):
        package_root = Path(__file__).parents[1] / "QickworkspaceV2"
        offenders = []
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name if node.names else None
                if module == "qick_workspace" or (
                    module and module.startswith("qick_workspace.")
                ):
                    offenders.append(f"{path}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_allxy_keeps_signal_out_of_y_sweep_axis(self):
        program = Mock()
        program.acquire.return_value = [[np.array([0.25, 0.0])]]
        experiment = AllXY({
            "reps": 1,
            "relax_delay": 0.1,
            "threshold": 0.0,
        })

        with patch(
            "QickworkspaceV2.experiments.characterization.allxy.AllXYProgram",
            return_value=program,
        ), patch(
            "QickworkspaceV2.experiments.characterization.allxy.tqdm",
            side_effect=lambda values, **_: values,
        ):
            result = experiment.run(py_avg=2)

        self.assertIsInstance(result, ExperimentData)
        self.assertIsNone(result.y_axis)
        self.assertEqual(result.raw_iq.shape, (21,))
        self.assertEqual(result.dataset_dims["iq"], ["x"])
        self.assertTrue(result.metadata["threshold_discrimination"])

    def test_power_rabi_chevron_returns_dimensioned_experiment_data(self):
        gains = np.linspace(0.01, 0.09, 9)
        iterations = np.array([1, 2])
        response = np.sinc((gains - 0.05) / 0.02) ** 2 + 0.1
        rows = [
            [[np.column_stack((response * scale, np.zeros_like(response)))]]
            for scale in (1.0, 1.2)
        ]
        program = Mock()
        program.acquire.side_effect = rows
        experiment = PowerRabiChevron({})
        experiment._build_scan_axes = Mock(return_value=(gains, iterations))
        experiment._create_program = Mock(return_value=program)

        with patch(
            "QickworkspaceV2.experiments.qubit_ge.aae.tqdm",
            side_effect=lambda values, **_: values,
        ), patch(
            "QickworkspaceV2.experiments.qubit_ge.aae.display"
        ), patch(
            "QickworkspaceV2.experiments.qubit_ge.aae.update_display"
        ), patch(
            "QickworkspaceV2.experiments.qubit_ge.aae.clear_output"
        ), patch("matplotlib.pyplot.show"):
            result = experiment.run(py_avg=2)

        self.assertIsInstance(result, ExperimentData)
        self.assertEqual(result.raw_iq.shape, (2, 9))
        self.assertEqual(result.dataset_dims["iq"], ["y", "x"])
        self.assertEqual(result.avg_count, 2)
        self.assertLess(result.scalar_result, 0.1)


if __name__ == "__main__":
    unittest.main()
