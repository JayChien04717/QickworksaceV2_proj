import unittest
from unittest.mock import Mock

import numpy as np

from QickworkspaceV2.plotter.liveplot import SoftwareAverageRunner


class LiveplotThresholdTests(unittest.TestCase):
    def test_software_average_keeps_complex_iq_without_threshold(self):
        prog = Mock()
        prog.acquire.side_effect = [
            [[np.array([[1.0, 2.0], [3.0, 4.0]])]],
            [[np.array([[3.0, 4.0], [5.0, 6.0]])]],
        ]
        runner = SoftwareAverageRunner(
            prog=prog, soc="soc", py_avg=2, iq_process="real"
        )

        values, interrupted, avg_count = runner.run(lambda *_: None)

        np.testing.assert_array_equal(
            values, np.array([2.0 + 3.0j, 4.0 + 5.0j])
        )
        self.assertTrue(np.iscomplexobj(values))
        self.assertFalse(interrupted)
        self.assertEqual(avg_count, 2)

    def test_software_average_passes_threshold_and_averages_real_values(self):
        prog = Mock()
        prog.acquire.side_effect = [
            [[np.array([0.0, 1.0])]],
            [[np.array([1.0, 1.0])]],
        ]
        updates = []
        runner = SoftwareAverageRunner(
            prog=prog,
            soc="soc",
            py_avg=2,
            iq_process="real",
            threshold=0.4,
        )

        values, interrupted, avg_count = runner.run(
            lambda index, data: updates.append((index, data.copy()))
        )

        self.assertEqual(prog.acquire.call_count, 2)
        prog.acquire.assert_called_with(
            "soc",
            rounds=1,
            progress=False,
            threshold=0.4,
        )
        np.testing.assert_array_equal(values, np.array([0.5, 1.0]))
        self.assertFalse(np.iscomplexobj(values))
        self.assertFalse(interrupted)
        self.assertEqual(avg_count, 2)
        self.assertEqual([index for index, _ in updates], [0, 1])


if __name__ == "__main__":
    unittest.main()
