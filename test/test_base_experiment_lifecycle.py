import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.core.base_experiment import BaseExperiment
from QickworkspaceV2.core.experiment_components import AcquisitionResult
from QickworkspaceV2.core.experiment_data import ExperimentData


class _RecordingAnalysis:
    plotted = False

    def run(self, result):
        result.metadata["analysis_ran"] = True
        return result

    def plot(self, result):
        type(self).plotted = True


class _LifecycleExperiment(BaseExperiment):
    EXPT_NAME = "lifecycle_test"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "MHz"
    LivePlot = False
    Analysis = _RecordingAnalysis

    def _create_program(self):
        self.created_program = object()
        return self.created_program

    def _extract_sweep_axis(self, prog):
        self.axis_program = prog
        return np.array([1.0, 2.0])

    def _acquire(self, prog, axes, ctx):
        self.acquire_program = prog
        self.acquire_context = ctx
        self.iqdata = np.array([1 + 2j, 3 + 4j])
        return AcquisitionResult(
            raw_iq=self.iqdata,
            avg_count=ctx.py_avg,
            analysis_data={
                "verification": {
                    "values": np.array([0.1, 0.2]),
                    "dims": ["x"],
                }
            },
        )


class BaseExperimentLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_runtime = (
            BaseExperiment._runtime.soc,
            BaseExperiment._runtime.soccfg,
            BaseExperiment._runtime.data_path,
            BaseExperiment._soc,
            BaseExperiment._soccfg,
            BaseExperiment._data_path,
        )
        BaseExperiment.setup(object(), object(), "/tmp")

    @classmethod
    def tearDownClass(cls):
        (
            BaseExperiment._runtime.soc,
            BaseExperiment._runtime.soccfg,
            BaseExperiment._runtime.data_path,
            BaseExperiment._soc,
            BaseExperiment._soccfg,
            BaseExperiment._data_path,
        ) = cls.previous_runtime

    def test_run_keeps_lifecycle_explicit_and_preserves_custom_acquisition(self):
        _RecordingAnalysis.plotted = False
        experiment = _LifecycleExperiment({"steps": 2})

        def build_result(**kwargs):
            return ExperimentData(experiment_id="test-id", **kwargs)

        with patch(
            "QickworkspaceV2.core.experiment_components.ExperimentData",
            side_effect=build_result,
        ):
            result = experiment.run(
                py_avg=3,
                liveplot=False,
                plot_analysis=True,
            )

        self.assertIs(experiment._last_prog, experiment.created_program)
        self.assertIs(experiment.axis_program, experiment.created_program)
        self.assertIs(experiment.acquire_program, experiment.created_program)
        self.assertEqual(experiment.acquire_context.py_avg, 3)
        np.testing.assert_array_equal(result.x_axis, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(
            result.analysis_data["verification"]["values"],
            np.array([0.1, 0.2]),
        )
        self.assertEqual(result.dataset_dims["iq"], ["x"])
        self.assertTrue(result.metadata["analysis_ran"])
        self.assertTrue(_RecordingAnalysis.plotted)
        self.assertEqual(result.config, {})
        self.assertIs(experiment.result, result)

    def test_save_labber_uses_experiment_config_yaml_when_provided(self):
        experiment = _LifecycleExperiment({"steps": 2})
        experiment._sweep_vals_x = np.array([1.0, 2.0])
        experiment._sweep_vals_y = None
        experiment.iqdata = np.array([1 + 2j, 3 + 4j])
        config_all = Mock()
        config_all.to_yaml.return_value = "formatted config"
        system_tool = types.ModuleType("QickworkspaceV2.tools.system_tool")
        system_tool.config_to_yaml = Mock()
        system_tool.get_next_filename_labber = Mock(
            return_value="/tmp/lifecycle-test.hdf5"
        )
        system_tool.hdf5_generator = Mock()

        with patch.dict(
            sys.modules,
            {"QickworkspaceV2.tools.system_tool": system_tool},
        ), patch("builtins.print"):
            experiment.saveLabber("Q1", config_all=config_all)

        config_all.to_yaml.assert_called_once_with(q_id="Q1")
        self.assertEqual(
            system_tool.hdf5_generator.call_args.kwargs["comment"],
            "formatted config",
        )

    def test_run_rejects_invalid_average_count_before_building_program(self):
        experiment = _LifecycleExperiment({"steps": 2})

        with self.assertRaisesRegex(ValueError, "py_avg must be at least 1"):
            experiment.run(py_avg=0, liveplot=False)

        self.assertFalse(hasattr(experiment, "created_program"))

    def test_run_normalizes_iq_process_alias(self):
        experiment = _LifecycleExperiment({"steps": 2})

        result = experiment.run(py_avg=1, iq_process="I", liveplot=False)

        self.assertEqual(result.metadata["iq_process"], "real")


if __name__ == "__main__":
    unittest.main()
