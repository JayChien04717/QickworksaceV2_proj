import unittest
from unittest.mock import Mock, call

from QickworkspaceV2.experiments.coherence.ramsey_reset import (
    ActiveResetRamseyProgram,
)


class ActiveResetRamseyTests(unittest.TestCase):
    def _program(self):
        prog = ActiveResetRamseyProgram.__new__(ActiveResetRamseyProgram)
        prog.soccfg = {"readouts": {0: {"tproc_ch": 2}}}
        prog.ro_chs = {0: {"length": 10}}
        for name in (
            "setup_resonator",
            "setup_qubit_gen",
            "add_loop",
            "setup_qb_pulse",
            "add_reg",
            "write_reg",
        ):
            setattr(prog, name, Mock())
        return prog

    def test_q1_feedback_threshold_and_direction(self):
        prog = self._program()
        cfg = {
            "ro_ch": 0,
            "steps": 11,
            "wait_time": 2.0,
            "virtual_detune": 0.1,
            "threshold": 6.549187014565937,
            "ro_length": 10,
            "reset_component": "I",
            "reset_excited_if": ">=",
        }

        prog._initialize(cfg)

        self.assertEqual(prog.ground_test, "<")
        self.assertEqual(prog.reset_threshold_raw, 65)
        prog.write_reg.assert_called_once_with("reset_threshold", 65)
        reset_calls = [
            item
            for item in prog.setup_qb_pulse.call_args_list
            if item.kwargs.get("name") == "reset_pi"
        ]
        self.assertEqual(len(reset_calls), 1)
        self.assertEqual(reset_calls[0].kwargs["gain_key"], "pi_gain_ge")

    def test_body_contains_enabled_conditional_reset_and_two_readouts(self):
        prog = ActiveResetRamseyProgram.__new__(ActiveResetRamseyProgram)
        prog.reset_component = "I"
        prog.ground_test = "<"
        for name in (
            "send_readoutconfig",
            "pulse",
            "delay_auto",
            "_readout",
            "wait_auto",
            "resync",
            "trigger",
            "read_and_jump",
            "label",
        ):
            setattr(prog, name, Mock())
        cfg = {
            "ro_ch": 0,
            "res_ch": 2,
            "qb_ch": 1,
            "trig_time": 0.5,
            "wait_time": 2.0,
            "read_wait": 5.0,
            "extra_delay": 0.1,
            "reset_post_delay": 0.05,
        }

        prog._body(cfg)

        self.assertEqual(prog._readout.call_count, 1)
        prog.wait_auto.assert_called_once_with(5.0, gens=True, ros=True)
        prog.resync.assert_called_once_with(0.05)
        prog.read_and_jump.assert_called_once_with(
            ro_ch=0,
            component="I",
            threshold="reset_threshold",
            test="<",
            label="AFTER_ACTIVE_RESET",
        )
        self.assertIn(
            call(ch=1, name="reset_pi", t=0), prog.pulse.call_args_list
        )
        prog.label.assert_called_once_with("AFTER_ACTIVE_RESET")


if __name__ == "__main__":
    unittest.main()