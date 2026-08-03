import unittest

import numpy as np

from QickworkspaceV2.analysis.qubit import RamseyEfAnalysis
from QickworkspaceV2.core.experiment_components import (
    AcquisitionResult,
    ResultBuilder,
    RunContext,
    SweepAxes,
)
from QickworkspaceV2.core.experiment_data import ExperimentData
from QickworkspaceV2.experiments.coherence.ramsey_ef import RamseyEf


class _EfExperiment:
    Analysis = RamseyEfAnalysis
    EXPT_NAME = "ramsey_ef_test"
    X_SAVE_NAME = "Delay"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6
    Y_SAVE_NAME = ""
    Y_SAVE_UNIT = ""
    Y_SAVE_SCALE = 1.0
    cfg = {
        "virtual_detune": 2.0,
        "qb_freq_ef": 4200.0,
        "unrelated": "not persisted",
    }
    fit_params = None
    fit_errors = None

    @staticmethod
    def _post_fit(_):
        return None

    @staticmethod
    def _apply_old_result(result, old_result):
        return None

    @staticmethod
    def _build_fit_result():
        return {}


class AnalysisContextTests(unittest.TestCase):
    def test_result_builder_persists_only_required_analysis_context(self):
        result = ResultBuilder().build(
            _EfExperiment(),
            AcquisitionResult(raw_iq=np.array([1 + 2j, 2 + 3j])),
            SweepAxes(x=np.array([0.0, 1.0]), y=None),
            RunContext(
                py_avg=1,
                iq_process="all",
                show_final_plot=False,
                liveplot=False,
                plot_analysis=False,
                kwargs={},
            ),
        )

        self.assertEqual(
            result.metadata["analysis_context"],
            {"virtual_detune": 2.0, "qb_freq_ef": 4200.0},
        )
        self.assertEqual(result.config, {})

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
