import unittest

import numpy as np

from QickworkspaceV2.analysis.qubit import RamseyEfAnalysis
from QickworkspaceV2.analysis.resonator import LorentzianAnalysis
from QickworkspaceV2.core.base_experiment import BaseExperiment
from QickworkspaceV2.core.experiment_components import AcquisitionResult
from QickworkspaceV2.core.experiment_data import ExperimentData, QualityFlag
from QickworkspaceV2.experiments.coherence.ramsey_ef import RamseyEf


class _ContextAnalysis:
    REQUIRED_CONFIG_KEYS = ("virtual_detune", "qb_freq_ef")

    def run(self, data):
        return data


class _EfExperiment(BaseExperiment):
    Analysis = _ContextAnalysis
    EXPT_NAME = "ramsey_ef_test"
    X_SAVE_NAME = "Delay"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6
    Y_SAVE_NAME = ""
    Y_SAVE_UNIT = ""
    Y_SAVE_SCALE = 1.0
    LivePlot = False

    def _create_program(self):
        return object()

    def _extract_sweep_axis(self, prog):
        return np.array([0.0, 1.0])

    def _acquire(self, prog, x_vals, y_vals, options):
        return AcquisitionResult(raw_iq=np.array([1 + 2j, 2 + 3j]))


class AnalysisContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_session = (
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
        ) = cls.previous_session

    def test_result_builder_persists_only_required_analysis_context(self):
        experiment = _EfExperiment({
            "virtual_detune": 2.0,
            "qb_freq_ef": 4200.0,
            "fit_channel": "real",
            "unrelated": "not persisted",
        })
        result = experiment.run(py_avg=1, liveplot=False)

        self.assertEqual(
            result.metadata["analysis_context"],
            {
                "fit_channel": "real",
                "virtual_detune": 2.0,
                "qb_freq_ef": 4200.0,
            },
        )
        self.assertEqual(result.config, {})

    def test_missing_fit_channel_uses_auto_instead_of_mapping_placeholder(self):
        analysis = LorentzianAnalysis()
        data = ExperimentData(
            raw_iq=np.array([1 + 0j, 2 + 0j]),
            x_axis=np.array([0.0, 1.0]),
            metadata={"analysis_context": {"fit_channel": {}}},
        )
        seen = []

        def fake_fit(x, y, fitparams=None):
            seen.append(np.asarray(y))
            return np.array([0.0, 1.0, 0.5, 0.2]), np.eye(4), fitparams

        analysis._fit_channel(data, fake_fit, lambda x, *p: np.zeros_like(x))
        self.assertEqual(data.metadata["fit_channel"], "abs")
        self.assertTrue(seen)

    def test_lorentzian_result_uses_scale_as_linewidth(self):
        analysis = LorentzianAnalysis()
        data = ExperimentData(
            raw_iq=np.array([1 + 0j, 2 + 0j]),
            x_axis=np.array([0.0, 1.0]),
        )
        popt = np.array([10.0, -4.0, 0.5, 0.25])
        analysis._fit_channel = lambda *args, **kwargs: (
            np.ones(2), popt, np.eye(4), "abs", 8.0
        )

        analysis._run(data)

        self.assertEqual(data.get_param("amplitude"), -4.0)
        self.assertEqual(data.get_param("offset"), 10.0)
        self.assertEqual(data.get_param("linewidth_MHz"), 0.5)

    def test_render_falls_back_to_raw_data_when_fit_failed(self):
        analysis = LorentzianAnalysis()
        data = ExperimentData(
            experiment_type="qubit_spec",
            raw_iq=np.array([1 + 0j, 2 + 1j]),
            x_axis=np.array([0.0, 1.0]),
            quality=QualityFlag.BAD,
            quality_message="fit did not converge",
        )
        calls = []
        analysis._show_raw = lambda result: calls.append(("raw", result))
        analysis.plot = lambda result: calls.append(("fit", result))

        analysis.render(data)

        self.assertEqual(calls, [("raw", data)])

    def test_render_keeps_bad_but_completed_fit_visible(self):
        analysis = LorentzianAnalysis()
        data = ExperimentData(
            raw_iq=np.array([1 + 0j, 2 + 1j]),
            x_axis=np.array([0.0, 1.0]),
            fit_params=np.array([0.0, 1.0, 0.5, 0.2]),
            fit_result={"f0_MHz": 0.5},
            quality=QualityFlag.BAD,
            quality_message="linewidth outside threshold",
        )
        calls = []
        analysis._show_raw = lambda result: calls.append(("raw", result))
        analysis.plot = lambda result: calls.append(("fit", result))

        analysis.render(data)

        self.assertEqual(calls, [("fit", data)])

    def test_raw_plot_trace_reduces_leading_dimensions(self):
        data = ExperimentData(
            raw_iq=np.array([[1 + 0j, 3 + 0j], [3 + 0j, 5 + 0j]]),
            x_axis=np.array([10.0, 20.0]),
        )

        x, iq, note = LorentzianAnalysis._raw_plot_trace(data)

        np.testing.assert_array_equal(x, [10.0, 20.0])
        np.testing.assert_array_equal(iq, [2.0 + 0j, 4.0 + 0j])
        self.assertIn("averaged over 2 traces", note)

    def test_ef_ramsey_correction_uses_ef_frequency(self):
        analysis = RamseyEfAnalysis()
        data = ExperimentData(
            raw_iq=np.array([1 + 0j, 0 + 1j]),
            x_axis=np.array([0.0, 1.0]),
            config={"qb_freq_ge": 3000.0},
            metadata={
                "analysis_context": {
                    "virtual_detune": 2.0,
                    "qb_freq_ef": 4200.0,
                }
            },
        )
        popt = np.array([1.0, 2.25, 0.0, 20.0, 0.0])
        analysis._fit_channel = lambda *args, **kwargs: (
            np.ones(2), popt, np.eye(5), "abs", 10.0
        )

        analysis._fit_decaysin(data)

        self.assertAlmostEqual(data.get_param("corrected_freq_MHz"), 4199.75)
        self.assertIs(RamseyEf.Analysis, RamseyEfAnalysis)


if __name__ == "__main__":
    unittest.main()
