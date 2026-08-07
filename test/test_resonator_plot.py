import unittest
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.analysis.resonator import ResonatorSpecAnalysis
from QickworkspaceV2.core.experiment_data import ExperimentData, QualityFlag


class ResonatorPlotTests(unittest.TestCase):
    def test_abcd_result_uses_framework_plot_with_four_requested_metrics(self):
        frequency = np.linspace(6715.0, 6720.0, 21)
        measured = 1.0 - 0.4 / (
            1 + 2j * (frequency - 6717.5) / 0.15
        )
        fitted = 1.0 - 0.39 / (
            1 + 2j * (frequency - 6717.51) / 0.148
        )
        data = ExperimentData(
            raw_iq=measured,
            x_axis=frequency,
            fit_result={
                "f_res[MHz]": (6717.51234, None),
                "kappa_MHz": (0.14842, None),
                "Qi": (123456, None),
                "Qc": (65432, None),
                "Ql": (42000, None),
            },
            analysis_data={
                "fit_curve": {"values": fitted, "dims": ["x"]}
            },
            quality=QualityFlag.GOOD,
        )
        figure = Mock()

        with patch(
            "QickworkspaceV2.plotter.plot_utils.plot_fit_result",
            return_value=figure,
        ) as plot_fit:
            returned = ResonatorSpecAnalysis().plot(data)

        self.assertIs(returned, figure)
        self.assertEqual(
            plot_fit.call_args.kwargs["result_text"],
            "f_res     = 6717.512 MHz\n"
            "linewidth = 0.148 MHz\n"
            "Qi        = 123,456\n"
            "Qc        = 65,432",
        )
        self.assertNotIn("Ql", plot_fit.call_args.kwargs["result_text"])
        self.assertEqual(
            plot_fit.call_args.kwargs["title"], "Resonator Spectroscopy"
        )
        self.assertEqual(plot_fit.call_args.kwargs["fit_channel"], "abs")


if __name__ == "__main__":
    unittest.main()
