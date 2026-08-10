import unittest
from unittest.mock import Mock

from QickworkspaceV2.core.base_program import BaseProgram
from QickworkspaceV2.core.qubit_pulse import QubitPulseMixin


class _ConcreteBaseProgram(BaseProgram):
    def _initialize(self, cfg):
        pass

    def _body(self, cfg):
        pass


def _config():
    return {
        "pulse_type": "arb",
        "qb_ch": 1,
        "qb_ch_ef": 2,
        "qb_phase": 10,
        "qb_gain_ge": 0.1,
        "qb_gain_ef": 0.2,
        "qb_freq_ge": 4000,
        "qb_freq_ef": 3800,
        "sigma_ge": 0.05,
        "sigma_ef": 0.06,
        "qb_length_ge": 0.2,
        "qb_flat_top_length_ge": 0.3,
        "qb_flat_top_length_ef": 0.35,
        "pi_gain_ge": 0.15,
        "drag_alpha": 0.4,
    }


def _program():
    program = QubitPulseMixin()
    program.soccfg = {"gens": {1: {"has_mixer": False}, 2: {"has_mixer": True}}}
    program.declare_gen = Mock()
    program.add_gauss = Mock()
    program.add_cosine = Mock()
    program.add_DRAG = Mock()
    program.add_pulse = Mock()
    return program


class QubitPulseTests(unittest.TestCase):
    def test_measure_accepts_custom_marker_pins(self):
        program = _ConcreteBaseProgram.__new__(_ConcreteBaseProgram)
        program.pulse = Mock()
        program.trigger = Mock()

        program.measure(
            {"res_ch": 2, "ro_ch": 0, "trig_time": 0.4},
            pins=[1],
        )

        program.pulse.assert_called_once_with(ch=2, name="res_pulse", t=0)
        program.trigger.assert_called_once_with(ros=[0], pins=[1], t=0.4)

    def test_cooling_body_allows_one_configured_channel(self):
        program = _ConcreteBaseProgram.__new__(_ConcreteBaseProgram)
        program.pulse = Mock()
        program.delay_auto = Mock()

        applied = program.cooling_body(
            {"cooling": True, "cool_ch1": 3},
            ring_down=0.25,
        )

        self.assertTrue(applied)
        program.pulse.assert_called_once_with(
            ch=3, name="cool_pulse1", t=0
        )
        program.delay_auto.assert_called_once_with(0.25, tag="Ring down")

    def test_qubit_generator_uses_qick_declare_gen_api(self):
        program = _program()
        cfg = _config()
        cfg.update(
            {
                "nqz_qb": 1,
                "qb_mixer": 500,
                "nqz_qb_ef": 2,
                "qb_mixer_ef": 600,
            }
        )

        program.setup_qubit_gen(cfg, "ge")
        program.setup_qubit_gen(cfg, "ef")

        self.assertEqual(
            program.declare_gen.call_args_list[0].kwargs,
            {"ch": 1, "nqz": 1},
        )
        self.assertEqual(
            program.declare_gen.call_args_list[1].kwargs,
            {"ch": 2, "nqz": 2, "mixer_freq": 600},
        )

    def test_arb_envelope_is_reused_and_gain_key_is_resolved(self):
        program = _program()
        cfg = _config()

        program.setup_qb_pulse(cfg, name="x180", gain_key="pi_gain_ge")
        program.setup_qb_pulse(cfg, name="y180", phase=90)

        program.add_gauss.assert_called_once_with(
            ch=1,
            name="env_ge_gauss",
            sigma=0.05,
            length=0.25,
            even_length=True,
        )
        self.assertEqual(program.add_pulse.call_args_list[0].kwargs["gain"], 0.15)
        self.assertEqual(program.add_pulse.call_args_list[1].kwargs["phase"], 90)

    def test_const_pulse_does_not_create_an_envelope(self):
        program = _program()

        program.setup_qb_pulse(
            _config(), pulse_type="const", gain_override=0, name="const"
        )

        program.add_gauss.assert_not_called()
        program.add_pulse.assert_called_once_with(
            ch=1,
            name="const",
            style="const",
            freq=4000,
            phase=10,
            gain=0,
            length=0.2,
        )

    def test_drag_type_forces_drag_envelope_and_registers_arb_pulse(self):
        program = _program()

        program.setup_qb_pulse(
            _config(), pulse_type="drag", shape="gauss", name="drag"
        )

        program.add_DRAG.assert_called_once_with(
            ch=1,
            name="env_ge_drag",
            sigma=0.05,
            length=0.25,
            delta=200,
            alpha=0.4,
            even_length=True,
        )
        self.assertEqual(program.add_pulse.call_args.kwargs["style"], "arb")
        self.assertEqual(
            program.add_pulse.call_args.kwargs["envelope"], "env_ge_drag"
        )

    def test_flat_top_ef_uses_transition_channel_frequency_gain_and_length(self):
        program = _program()

        program.setup_qb_pulse(
            _config(), prefix="ef", pulse_type="flat_top", name="ef"
        )

        program.add_gauss.assert_called_once_with(
            ch=2,
            name="env_ef_gauss",
            sigma=0.06,
            length=0.3,
            even_length=True,
        )
        program.add_pulse.assert_called_once_with(
            ch=2,
            name="ef",
            style="flat_top",
            freq=3800,
            phase=10,
            gain=0.2,
            envelope="env_ef_gauss",
            length=0.35,
        )

    def test_unknown_pulse_type_fails_explicitly(self):
        program = _program()

        with self.assertRaisesRegex(ValueError, "Unknown qubit pulse type"):
            program.setup_qb_pulse(_config(), pulse_type="mystery")

        program.add_pulse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
