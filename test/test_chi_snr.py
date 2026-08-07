import unittest

import numpy as np

from QickworkspaceV2.analysis.resonator import DispersiveShiftAnalysis
from QickworkspaceV2.core.experiment_data import ExperimentData
from QickworkspaceV2.experiments.resonator.chi import (
    _decode_frequency_shots,
    _frequency_snr,
)


class ChiSNRTests(unittest.TestCase):
    def test_decode_accepts_both_qick_loop_orders(self):
        shots, frequencies = 7, 5
        expected = np.arange(2 * shots * frequencies).reshape(2, shots, frequencies)
        iq = np.stack((expected.real, expected.imag), axis=-1)
        np.testing.assert_allclose(
            _decode_frequency_shots(iq, shots, frequencies), expected
        )
        np.testing.assert_allclose(
            _decode_frequency_shots(iq.transpose(0, 2, 1, 3), shots, frequencies),
            expected,
        )

    def test_dispersive_analysis_uses_empirical_points_without_fit(self):
        frequency = np.linspace(100.0, 106.0, 7)
        traces = np.full((2, 7), 5.0 + 0.0j)
        traces[0, 1:4] = [4.0, 1.0, 4.0]
        traces[1, 3:6] = [4.0, 1.0, 4.0]
        result = ExperimentData(raw_iq=traces, x_axis=frequency)

        DispersiveShiftAnalysis().run(result)

        self.assertEqual(result["f_res_g_MHz"], 102.0)
        self.assertEqual(result["f_res_e_MHz"], 104.0)
        self.assertEqual(result["chi_MHz"], 1.0)
        self.assertEqual(
            result.metadata["chi_method"],
            "empirical_smoothed_magnitude_minimum",
        )
    def test_snr_selects_largest_state_separation(self):
        rng = np.random.default_rng(8)
        shots, frequencies = 2000, 4
        noise = 0.2
        ground = noise * (
            rng.normal(size=(shots, frequencies))
            + 1j * rng.normal(size=(shots, frequencies))
        )
        separation = np.array([0.2, 0.6, 1.5, 0.8])
        excited = ground + separation
        means, variance, snr = _frequency_snr(np.stack((ground, excited)))

        self.assertEqual(means.shape, (2, frequencies))
        self.assertEqual(variance.shape, (frequencies,))
        self.assertEqual(int(np.argmax(snr)), 2)
        self.assertTrue(np.all(np.isfinite(snr)))


if __name__ == "__main__":
    unittest.main()