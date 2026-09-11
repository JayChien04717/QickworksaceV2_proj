"""
QubitGE/rabi — s004: Time Rabi + s005: Power Rabi + s005b: Power Rabi with reset.
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment, SweepAxis
from ...analysis.qubit import PowerRabiAnalysis, TimeRabiAnalysis



class TimeRabiProgram(BaseProgram):
    """QICK program for time Rabi: sweeps flat-top pulse length."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("lenloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class TimeRabi(BaseExperiment):
    """Time Rabi (ge): sweeps flat-top pulse length, fits decaying sinusoid."""

    EXPT_NAME = "s004_time_rabi_ge"
    TAG = "TimeRabi"
    X_LABEL = "Pulse Length (us)"
    TITLE_PREFIX = "Qubit Time Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_flat_top_length_ge"]
    X_SAVE_NAME = "Pulse Length"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    Analysis = TimeRabiAnalysis
    PROGRAM = TimeRabiProgram
    X_AXIS = SweepAxis.pulse("qb_pulse", "length")




class PowerRabiProgram(BaseProgram):
    """QICK program for power Rabi: sweeps qubit drive gain."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabi(BaseExperiment):
    """Power Rabi (ge): sweeps gain, fits sinusoid for pi and pi/2 gains."""

    EXPT_NAME = "s005_power_rabi_ge"
    TAG = "PowerRabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    Analysis = PowerRabiAnalysis
    PROGRAM = PowerRabiProgram
    X_AXIS = SweepAxis.pulse("qb_pulse", "gain")




class PowerRabiResetProgram(BaseProgram):
    """Power Rabi with active-reset cooling before each shot."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class PowerRabiReset(PowerRabi):
    """Power Rabi with active reset (s005b)."""

    EXPT_NAME = "s005b_power_rabi_reset_ge"
    TAG = "PowerRabi"
    TITLE_PREFIX = "Qubit Power Rabi ge (Reset)"
    PROGRAM = PowerRabiResetProgram
