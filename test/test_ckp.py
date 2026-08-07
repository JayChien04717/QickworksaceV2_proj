import unittest

import numpy as np

from QickworkspaceV2.core.experiment_data import ExperimentData
from QickworkspaceV2.experiments.resonator.ckp import (
    CKPAnalysis,
    _decode_ckp_map,
    _joint_ckp_model,
)


class CKPTests(unittest.TestCase):
    def test_decoder_accepts_both_qick_loop_orders(self):
        expected = np.arange(35, dtype=float).reshape(5, 7)
        np.testing.assert_array_equal(_decode_ckp_map(expected, 7, 5), expected)
        np.testing.assert_array_equal(
            _decode_ckp_map(expected.T, 7, 5), expected
        )
        square_qick_order = np.arange(16, dtype=float).reshape(4, 4)
        np.testing.assert_array_equal(
            _decode_ckp_map(square_qick_order, 4, 4),
            square_qick_order.T,
        )

    def test_analysis_recovers_synthetic_parameters(self):
        resonator_frequency = np.linspace(6692.0, 6708.0, 61)
        qubit_frequency = np.linspace(4965.0, 5002.0, 181)
        parameters = (5000.0, 6700.0, -1.6, 4.2, -22.0)
        magnitude = np.empty(
            (2, qubit_frequency.size, resonator_frequency.size)
        )
        for state in range(2):
            center = _joint_ckp_model(
                (
                    np.full(resonator_frequency.shape, state),
                    resonator_frequency,
                ),
                *parameters,
            )
            magnitude[state] = 0.03 + 0.82 / (
                1 + ((qubit_frequency[:, None] - center) / 0.45) ** 2
            )
        phase = 0.3 + 0.002 * (resonator_frequency[None, None, :] - 6700)
        raw_iq = magnitude * np.exp(1j * phase)

        result = ExperimentData(
            raw_iq=raw_iq,
            x_axis=resonator_frequency,
            y_axis=qubit_frequency,
            metadata={
                "analysis_context": {
                    "ckp_min_slice_contrast": 0.02,
                    "ckp_res_gain": 0.2,
                    "res_freq_ge": 6700.0,
                }
            },
        )
        CKPAnalysis().run(result)

        self.assertTrue(result.is_good(), result.quality_message)
        self.assertAlmostEqual(result["chi_MHz"], parameters[2], places=3)
        self.assertAlmostEqual(result["kappa_MHz"], parameters[3], places=3)
        self.assertAlmostEqual(
            result["nbar_resonant"],
            parameters[4] / (2 * parameters[2]),
            places=3,
        )
        self.assertGreater(result["joint_fit_r2"], 0.999)
        self.assertEqual(result.metadata["fit_channel"], "pca")


if __name__ == "__main__":
    unittest.main()
