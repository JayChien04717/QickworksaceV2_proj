import unittest
from unittest.mock import Mock

import numpy as np

from QickworkspaceV2.experiments.cryoscope._common import (
    CryoscopeProgramBase,
    generator_sample_period_ns,
    make_zero_padded_rectangle,
    pad_normalized_envelope,
    quantize_envelope,
)
from QickworkspaceV2.experiments.cryoscope.filter_design import (
    apply_inverse_fir,
    design_inverse_fir,
    merge_xy_segments,
    predict_corrected_output,
    trace_from_xy,
)


class _SocCfg(dict):
    def get_maxv(self, ch):
        return 32767


class CryoscopeWaveformTests(unittest.TestCase):
    def setUp(self):
        self.soccfg = _SocCfg(
            gens={3: {"samps_per_clk": 16, "f_fabric": 599.04}}
        )

    def test_short_rectangle_is_trailing_padded_to_three_clocks(self):
        waveform = make_zero_padded_rectangle(
            self.soccfg, 3, active_samples=5, amplitude=0.5
        )

        self.assertEqual(waveform.shape, (48,))
        np.testing.assert_array_equal(waveform[:5], np.full(5, 0.5))
        np.testing.assert_array_equal(waveform[5:], np.zeros(43))
        self.assertAlmostEqual(
            generator_sample_period_ns(self.soccfg, 3),
            1000 / (599.04 * 16),
        )

    def test_zero_point_and_quantization_are_well_defined(self):
        waveform = make_zero_padded_rectangle(
            self.soccfg, 3, active_samples=0
        )

        self.assertEqual(waveform.shape, (48,))
        self.assertFalse(np.any(waveform))
        quantized = quantize_envelope(self.soccfg, 3, [0.0, 0.5, -1.0])
        np.testing.assert_array_equal(quantized, [0, 16384, -32767])

    def test_invalid_envelopes_are_rejected_before_upload(self):
        with self.assertRaisesRegex(ValueError, "peak"):
            pad_normalized_envelope(self.soccfg, 3, [0.0, 1.01])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            pad_normalized_envelope(self.soccfg, 3, [])

    def test_flux_generator_uses_current_declare_gen_api(self):
        program = object.__new__(CryoscopeProgramBase)
        program.soccfg = {
            "gens": {3: {"has_mixer": True}},
        }
        program.setup_resonator = Mock()
        program.setup_qubit_gen = Mock()
        program.setup_standard_gates = Mock()
        program.declare_gen = Mock()
        program._add_flux_pulse = Mock()
        cfg = {
            "flux_ch": 3,
            "qb_ch": 1,
            "res_ch": 2,
            "nqz_flux": 2,
            "flux_mixer": 125.0,
        }

        CryoscopeProgramBase._initialize(program, cfg)

        program.declare_gen.assert_called_once_with(
            ch=3, nqz=2, mixer_freq=125.0
        )
        program._add_flux_pulse.assert_called_once_with(cfg)


class CryoscopeFilterTests(unittest.TestCase):
    def test_xy_trace_and_regularized_inverse_are_finite(self):
        time_ns = np.linspace(0.0, 100.0, 101)
        phase = 2 * np.pi * 0.01 * time_ns
        trace = trace_from_xy(
            time_ns, np.cos(phase), np.sin(phase), tail_points=10
        )
        np.testing.assert_allclose(trace.detuning_mhz[10:-10], 10.0, atol=1e-8)

        step = 1 - np.exp(-np.linspace(0.0, 1.0, 101) / 0.08)
        design = design_inverse_fir(step, taps=16)
        target = np.r_[np.ones(32), np.zeros(8)]
        corrected = predict_corrected_output(
            apply_inverse_fir(target, design), step
        )
        self.assertEqual(design.taps.shape, (16,))
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_segment_merge_sorts_and_averages_duplicate_times(self):
        time_ns, x, y = merge_xy_segments(
            ([1.0, 2.0], [2.0, 4.0], [1.0, 3.0]),
            ([0.0, 1.0], [0.0, 4.0], [0.0, 5.0]),
        )
        np.testing.assert_array_equal(time_ns, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(x, [0.0, 3.0, 4.0])
        np.testing.assert_array_equal(y, [0.0, 3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
