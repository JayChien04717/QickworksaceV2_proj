import unittest
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.analysis.rb import RBAnalysis
from QickworkspaceV2.core.experiment_data import ExperimentData
from QickworkspaceV2.experiments.characterization.rb import (
    RandomizedBenchmarking,
    _rb_sample_matrix,
)


def _rb_experiment(threshold):
    rb = RandomizedBenchmarking.__new__(RandomizedBenchmarking)
    rb.cfg = {
        "name": "Q1",
        "reps": 10,
        "relax_delay": 1.0,
        "threshold": threshold,
    }
    rb.soc = "soc"
    rb.soccfg = "soccfg"
    rb.Analysis = None
    rb.x = None
    rb.rb_result = None
    rb._number_sample = None
    rb._interleaved = None
    rb._iq_process = "abs"
    rb._threshold_discrimination = False
    return rb


class RBThresholdTests(unittest.TestCase):
    @patch("QickworkspaceV2.experiments.characterization.rb.RBProgram")
    def test_threshold_acquisition_uses_real_i_population(self, rb_program):
        programs = [Mock(), Mock()]
        for index, program in enumerate(programs):
            population = 0.2 + 0.6 * index
            program.acquire.return_value = [[np.array([population, 99.0])]]
        rb_program.side_effect = programs

        rb = _rb_experiment(threshold=0.4)
        result = rb.run(
            py_avg=1,
            max_circuit_depth=4,
            delta_clifford=2,
            number_sample=1,
        )

        for program in programs:
            program.acquire.assert_called_once_with(
                "soc", rounds=1, progress=False, threshold=0.4
            )
        np.testing.assert_allclose(result.raw_iq, [[0.2], [0.8]])
        self.assertFalse(np.iscomplexobj(result.raw_iq))
        self.assertEqual(result.metadata["iq_process"], "real")
        self.assertEqual(result.metadata["threshold"], 0.4)
        self.assertTrue(result.metadata["threshold_discrimination"])
        self.assertEqual(rb._iq_process, "real")
        self.assertTrue(rb._threshold_discrimination)
        np.testing.assert_allclose(result.y_axis, [0.8, 0.2])
        self.assertEqual(result.metadata["reported_population"], "ground_survival")

    def test_retained_scalar_q_placeholder_is_not_averaged_into_population(self):
        raw = np.array([[[0.1, 0.0], [0.4, 0.0]]])

        survival = _rb_sample_matrix(
            raw, n_depths=1, n_samples=2, iq_process="real",
            threshold_discrimination=True,
        )

        np.testing.assert_allclose(survival, [[0.9, 0.6]])
    def test_misspelled_threshold_key_fails_explicitly(self):
        rb = _rb_experiment(threshold=None)
        rb.cfg.pop("threshold")
        rb.cfg["theshold"] = 0.4

        with self.assertRaisesRegex(KeyError, "use 'threshold'"):
            rb._get_readout_threshold()
    @patch("QickworkspaceV2.experiments.characterization.rb.error_fit_err", return_value=0.0)
    @patch("QickworkspaceV2.experiments.characterization.rb.rb_error", return_value=0.0)
    @patch("QickworkspaceV2.experiments.characterization.rb.rb_func", return_value=np.ones(400))
    @patch("QickworkspaceV2.experiments.characterization.rb.fitrb")
    def test_plot_flattens_retained_readout_dimensions_for_sem(
        self, fitrb, _rb_func, _rb_error, _error_fit_err
    ):
        fitrb.return_value = (
            np.array([0.99, 0.5, 0.1]),
            np.diag([0.01, 0.01, 0.01]),
        )
        rb = _rb_experiment(threshold=None)
        rb.x = np.arange(1, 11)
        rb._number_sample = 3
        rb.rb_result = np.arange(60, dtype=float).reshape(10, 3, 2)
        ax = Mock()

        rb.plot(ax=ax, show_individual=True)

        yerr = ax.errorbar.call_args.kwargs["yerr"]
        self.assertEqual(yerr.shape, (10,))
        self.assertEqual(ax.scatter.call_count, 4)  # three samples plus the mean
    @patch("QickworkspaceV2.tools.system_tool.config_to_yaml", return_value="cfg")
    @patch("QickworkspaceV2.tools.system_tool.get_next_filename_labber", return_value="rb.hdf5")
    @patch("QickworkspaceV2.tools.system_tool.hdf5_generator")
    def test_save_labber_writes_sample_by_depth_matrix(
        self, hdf5_generator, _get_filename, _config_to_yaml
    ):
        rb = _rb_experiment(threshold=None)
        rb.x = np.arange(1, 11)
        rb._number_sample = 5
        rb.rb_result = np.arange(100, dtype=float).reshape(10, 5, 2)
        rb.result = None

        rb.saveLabber("Q1")

        saved = hdf5_generator.call_args.kwargs["z_info"]["values"]
        self.assertEqual(saved.shape, (5, 10))
        expected = np.abs(np.asarray(rb.rb_result)).mean(axis=2).T
        np.testing.assert_allclose(saved, expected)
    @patch("QickworkspaceV2.tools.fitting.fitrb")
    @patch("QickworkspaceV2.tools.fitting.rb_func")
    def test_analysis_marks_threshold_population_as_real(
        self, rb_func, fitrb
    ):
        fitrb.return_value = (
            np.array([0.99, 0.5, 0.1]),
            np.diag([0.01, 0.01, 0.01]),
        )
        rb_func.return_value = np.array([0.8, 0.7])
        data = ExperimentData(
            experiment_type="rb",
            raw_iq=np.array([[0.8], [0.7]]),
            x_axis=np.array([1.0, 3.0]),
            metadata={"threshold_discrimination": True},
        )

        RBAnalysis().run(data)

        self.assertEqual(data.metadata["fit_channel"], "real")
        np.testing.assert_allclose(
            data.analysis_data["fit_input"]["values"], [0.2, 0.3]
        )


if __name__ == "__main__":
    unittest.main()
