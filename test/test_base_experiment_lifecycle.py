import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.core.base_experiment import BaseExperiment
from QickworkspaceV2.core.experiment_components import AcquisitionResult, SweepAxis
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

    def _acquire(self, prog, x_vals, y_vals, options):
        self.acquire_program = prog
        self.acquire_options = options
        self.iqdata = np.array([1 + 2j, 3 + 4j])
        return AcquisitionResult(
            raw_iq=self.iqdata,
            avg_count=options["py_avg"],
            analysis_data={
                "verification": {
                    "values": np.array([0.1, 0.2]),
                    "dims": ["x"],
                }
            },
        )


class _DeclaredProgram:
    def __init__(self, soccfg, *, reps, final_delay, cfg):
        self.constructor = (soccfg, reps, final_delay, cfg)

    def get_pulse_param(self, name, parameter, *, as_array):
        values = {
            ("readout", "freq"): np.array([100.0, 101.0]),
            ("readout", "gain"): np.array([0.01, 0.02, 0.03]),
        }
        return values[(name, parameter)]


class _DeclarativeExperiment(BaseExperiment):
    EXPT_NAME = "declarative_test"
    LivePlot = False
    PROGRAM = _DeclaredProgram
    X_AXIS = SweepAxis.pulse("readout", "freq")
    Y_AXIS = SweepAxis.pulse("readout", "gain")

    def _acquire(self, prog, x_vals, y_vals, options):
        self.acquired_axes = (x_vals, y_vals)
        return AcquisitionResult(
            raw_iq=np.ones((len(y_vals), len(x_vals)), dtype=complex),
            avg_count=options["py_avg"],
        )


class BaseExperimentLifecycleTests(unittest.TestCase):
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

    def test_run_keeps_lifecycle_explicit_and_preserves_custom_acquisition(self):
        _RecordingAnalysis.plotted = False
        experiment = _LifecycleExperiment({"steps": 2})

        def build_result(**kwargs):
            return ExperimentData(experiment_id="test-id", **kwargs)

        with patch(
            "QickworkspaceV2.core.base_experiment.ExperimentData",
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
        self.assertEqual(experiment.acquire_options["py_avg"], 3)
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

    def test_declarative_program_and_two_axes_are_the_fixed_interface(self):
        cfg = {"reps": 4, "relax_delay": 0.25, "steps": 2}
        experiment = _DeclarativeExperiment(cfg)

        result = experiment.run(py_avg=2)

        self.assertEqual(experiment._last_prog.constructor, (
            experiment.soccfg, 4, 0.25, cfg
        ))
        np.testing.assert_array_equal(result.x_axis, [100.0, 101.0])
        np.testing.assert_array_equal(result.y_axis, [0.01, 0.02, 0.03])
        self.assertEqual(result.dataset_dims["iq"], ["y", "x"])


if __name__ == "__main__":
    unittest.main()
