"""Hardware-independent tests for broadband resonator spectroscopy."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.core.base_experiment import BaseExperiment
from QickworkspaceV2.core.experiment_components import AcquisitionResult
from QickworkspaceV2.experiments.resonator import (
    BroadbandResonatorSpec,
    ResonatorSpec,
    ResonatorSpecProgram,
)


class BroadbandResonatorSpecTests(unittest.TestCase):
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

    def test_uses_existing_program_and_automatically_plots_resonators(self):
        experiment = BroadbandResonatorSpec({"steps": 3})
        program = object()
        frequencies = np.array([6700.0, 6850.0, 7000.0])
        fitted = np.array([6742.5, 6811.0, 6920.25, 6977.75])
        figure, axes = Mock(), Mock()

        experiment._create_program = Mock(return_value=program)
        experiment._extract_sweep_axis = Mock(return_value=frequencies)
        experiment._acquire = Mock(
            return_value=AcquisitionResult(
                raw_iq=np.ones(3, dtype=complex),
                avg_count=2,
            )
        )

        experiment.fit_n_resonators = Mock(return_value=(fitted, {}))
        with patch(
            "QickworkspaceV2.experiments.resonator.broadband_res_spec."
            "_plot_n_resonators",
            return_value=(figure, axes, fitted * 1e6),
        ) as plot_n:
            result = experiment.run(
                py_avg=2,
                count=4,
                show=False,
                solve_type="hm",
                detection_options={"use_phase_reference": False},
            )

        self.assertIs(experiment.PROGRAM, ResonatorSpecProgram)
        experiment.fit_n_resonators.assert_called_once_with(
            count=4,
            use_phase_reference=False,
        )
        plot_n.assert_called_once()
        plot_args, plot_kwargs = plot_n.call_args
        np.testing.assert_array_equal(plot_args[0], frequencies * 1e6)
        np.testing.assert_array_equal(plot_args[1], result.raw_iq)
        np.testing.assert_array_equal(plot_args[2], fitted * 1e6)
        self.assertEqual(plot_kwargs, {"y_mode": "abs", "show": False})
        np.testing.assert_array_equal(experiment.resonator_freqs, fitted)
        self.assertIs(experiment.resonator_figure, figure)
        self.assertIs(experiment.resonator_axes, axes)
        self.assertIn(figure, result.figures)
        self.assertNotIn(
            "solve_type",
            experiment._acquire.call_args.args[3]["kwargs"],
        )

    def test_multi_resonator_helpers_belong_only_to_broadband_experiment(self):
        self.assertNotIn("fit_n_resonators", ResonatorSpec.__dict__)
        self.assertNotIn("plot_n_resonators", ResonatorSpec.__dict__)
        self.assertIn("fit_n_resonators", BroadbandResonatorSpec.__dict__)
        self.assertIn("plot_n_resonators", BroadbandResonatorSpec.__dict__)

    def test_labber_filename_and_comment_do_not_select_a_qubit(self):
        experiment = BroadbandResonatorSpec({"steps": 2})
        experiment._sweep_vals_x = np.array([6700.0, 7000.0])
        experiment._sweep_vals_y = None
        experiment.iqdata = np.ones(2, dtype=complex)

        config_all = Mock()
        config_all.to_yaml.return_value = "full system config"
        system_tool = types.ModuleType("QickworkspaceV2.tools.system_tool")
        system_tool.config_to_yaml = Mock()
        system_tool.get_next_filename_labber = Mock(
            return_value="/tmp/broadband_resonator_spectrum_001.hdf5"
        )
        system_tool.hdf5_generator = Mock(
            return_value="/tmp/broadband_resonator_spectrum_001.hdf5"
        )

        with patch.dict(
            sys.modules,
            {"QickworkspaceV2.tools.system_tool": system_tool},
        ), patch("builtins.print"):
            experiment.saveLabber("Q7", config_all=config_all)

        system_tool.get_next_filename_labber.assert_called_once_with(
            "/tmp",
            "broadband_resonator_spectrum",
            None,
        )
        config_all.to_yaml.assert_called_once_with(q_id=None)
        self.assertEqual(
            system_tool.hdf5_generator.call_args.kwargs["comment"],
            "full system config",
        )


if __name__ == "__main__":
    unittest.main()
