"""Tests for multi-resonator detection and sub-bin fitting."""

import unittest

import numpy as np

from QickworkspaceV2.tools.fit_n_res import (
    detect_resonators,
    fit_n_resonators,
    phase_reference_candidates,
    plot_n_resonators,
)


class FitNResonatorsTest(unittest.TestCase):
    """Verify the hardware-independent multi-resonator fitting helpers."""

    @staticmethod
    def _synthetic_trace():
        frequency = np.linspace(4.0e9, 6.0e9, 4001)
        resonators = np.array([4.35e9, 5.02e9, 5.71e9])
        magnitude = np.ones_like(frequency)
        for center, depth, width in zip(
            resonators,
            (0.20, 0.32, 0.25),
            (5.0e6, 8.0e6, 6.0e6),
        ):
            magnitude -= depth / (1.0 + ((frequency - center) / width) ** 2)
        delay = 24e-9
        s21 = magnitude * np.exp(-2j * np.pi * frequency * delay)
        return frequency, s21, resonators

    def test_detects_requested_resonators(self):
        frequency, s21, expected = self._synthetic_trace()

        indices, properties, smooth_dips = detect_resonators(
            frequency,
            s21,
            count=3,
            use_phase_reference=False,
        )

        np.testing.assert_allclose(frequency[indices], expected, atol=0.6e6)
        self.assertEqual(indices.shape, (3,))
        self.assertEqual(smooth_dips.shape, frequency.shape)
        self.assertIn("scores", properties)

    def test_returns_sub_bin_frequency_estimates(self):
        frequency, s21, expected = self._synthetic_trace()

        fitted, details = fit_n_resonators(
            frequency,
            s21,
            count=3,
            use_phase_reference=False,
        )

        np.testing.assert_allclose(fitted, expected, atol=0.1e6)
        np.testing.assert_array_equal(details["fitted_freqs_hz"], fitted)
        self.assertEqual(details["dip_magnitudes"].shape, (3,))

    def test_phase_helper_estimates_electrical_delay(self):
        frequency, s21, _ = self._synthetic_trace()

        _, _, _, delay = phase_reference_candidates(frequency, s21)

        self.assertAlmostEqual(delay, 24e-9, delta=0.5e-9)

    def test_rejects_non_increasing_frequency(self):
        frequency, s21, _ = self._synthetic_trace()

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            fit_n_resonators(frequency[::-1], s21[::-1], count=3)

    def test_plots_fitted_resonators(self):
        frequency, s21, fitted = self._synthetic_trace()

        figure, axes, plotted = plot_n_resonators(
            frequency,
            s21,
            fitted,
            show=False,
        )

        np.testing.assert_array_equal(plotted, fitted)
        self.assertEqual(axes.get_xlabel(), "Frequency (MHz)")
        self.assertEqual(len(axes.lines), 4)
        figure.clear()


if __name__ == "__main__":
    unittest.main()
