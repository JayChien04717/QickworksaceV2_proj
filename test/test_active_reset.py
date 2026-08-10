import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np

from QickworkspaceV2.experiments.qubit_ge.rabi_reset import (
    ActiveResetRabi,
    ActiveResetRabiProgram,
)


class ActiveResetTests(unittest.TestCase):
    def test_threshold_acquisition_drops_q_placeholder_for_both_reads(self):
        pre = np.array([0.2, 0.7, 0.4])
        post = np.array([0.2, 0.3, 0.25])
        channel = np.stack((
            np.column_stack((pre, np.zeros_like(pre))),
            np.column_stack((post, np.zeros_like(post))),
        ))
        prog = Mock()
        prog.reset_threshold_normalized = 0.4
        prog.reset_component = "I"
        prog.ground_test = "<"
        prog.reset_threshold_raw = 1234
        prog.reset_readout_length = 10
        prog.reset_mode = "conditional"
        prog.acquire.return_value = [channel]
        expt = ActiveResetRabi.__new__(ActiveResetRabi)
        expt.soc = "soc"

        result = expt._acquire(
            prog, axes=None, ctx=SimpleNamespace(py_avg=3)
        )

        np.testing.assert_allclose(expt.pre_reset_population, pre)
        np.testing.assert_allclose(expt.post_reset_population, post)
        self.assertEqual(expt.pre_reset_population.shape, (3,))
        self.assertEqual(expt.post_reset_population.shape, (3,))
        prog.acquire.assert_called_once_with(
            "soc", rounds=3, threshold=0.4, angle=0.0, progress=True
        )
        np.testing.assert_allclose(result.raw_iq, pre)
        np.testing.assert_allclose(
            result.raw_data["readouts"], np.stack((pre, post))
        )
        self.assertEqual(result.dataset_dims["readouts"], ["readout", "x"])
        self.assertEqual(
            result.axes["readout"]["values"], ["pre_reset", "post_reset"]
        )

    def test_feedback_threshold_is_normalized_threshold_times_length(self):
        prog = ActiveResetRabiProgram.__new__(ActiveResetRabiProgram)
        prog.soccfg = {
            "readouts": {
                0: {"tproc_ch": 2, "ro_type": "axis_dyn_readout_v1"}
            },
        }
        prog.ro_chs = {0: {"length": 10, "ro_config": {"f_int": 123}}}
        prog.setup_resonator = Mock()
        prog.setup_qubit_gen = Mock()
        prog.add_loop = Mock()
        prog.setup_qb_pulse = Mock()
        prog._ro_offset = Mock(return_value=np.array([1.25, -0.5]))
        prog.add_reg = Mock()
        prog.write_reg = Mock()
        cfg = {
            "ro_ch": 0,
            "steps": 2,
            "qb_gain_ge": 0.1,
            "threshold": 0.5,
            "ro_length": 10,
            "reset_component": "I",
            "reset_excited_if": ">=",
        }

        prog._initialize(cfg)

        self.assertEqual(prog.ground_test, "<")
        self.assertEqual(prog.reset_threshold_raw, 5)
        prog._ro_offset.assert_not_called()
        prog.write_reg.assert_called_once_with("reset_threshold", 5)

    def _body_program(self, mode):
        prog = ActiveResetRabiProgram.__new__(ActiveResetRabiProgram)
        prog.reset_mode = mode
        prog.reset_component = "I"
        prog.ground_test = "<"
        for name in (
            "send_readoutconfig", "pulse", "delay_auto", "_readout",
            "wait_auto", "resync", "read_and_jump", "label", "trigger",
        ):
            setattr(prog, name, Mock())
        return prog

    def _body_cfg(self):
        return {
            "ro_ch": 0,
            "res_ch": 2,
            "qb_ch": 1,
            "trig_time": 0.5,
            "read_wait": 5.0,
            "extra_delay": 0.1,
            "reset_post_delay": 0.05,
        }

    def test_conditional_mode_executes_feedback_reset(self):
        prog = self._body_program("conditional")
        prog._body(self._body_cfg())

        prog.wait_auto.assert_called_once_with(5.0, gens=True, ros=True)
        prog.resync.assert_called_once_with(0.05)
        prog.read_and_jump.assert_called_once()
        self.assertIn(call(ch=1, name="reset_pi", t=0), prog.pulse.call_args_list)
        prog.label.assert_called_once_with("AFTER_ACTIVE_RESET")
        self.assertEqual(prog._readout.call_count, 1)

    def test_never_mode_uses_equal_timing_zero_gain_slot(self):
        prog = self._body_program("never")
        prog._body(self._body_cfg())

        prog.read_and_jump.assert_not_called()
        self.assertIn(call(ch=1, name="reset_idle", t=0), prog.pulse.call_args_list)
        self.assertNotIn(call(ch=1, name="reset_pi", t=0), prog.pulse.call_args_list)
        self.assertEqual(prog._readout.call_count, 2)

if __name__ == "__main__":
    unittest.main()
