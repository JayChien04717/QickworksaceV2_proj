"""
Resonator/res_punchout — s002b: Resonator punchout (2D gain × frequency).
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment, SweepAxis
from ...analysis.resonator import ResonatorPunchoutAnalysis


class PunchoutProgram(BaseProgram):
    """QICK program for resonator punchout: 2D sweep over gain and frequency."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.add_loop("gainloop", cfg["g_steps"])
        self.add_loop("freqloop", cfg["f_steps"])

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        self.measure(cfg)


class Punchout(BaseExperiment):
    """
    Resonator punchout: 2D sweep over gain and frequency.

    Maps power-dependent dispersive shift by varying resonator drive power
    (outer loop) and frequency (inner loop).
    """

    EXPT_NAME = "s002b_res_ge_punchout"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "DAC Gains"
    TITLE_PREFIX = "Resonator Punchout"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "DAC Gains"
    Y_SAVE_UNIT = "a.u."
    Y_SAVE_SCALE = 1.0

    Analysis = ResonatorPunchoutAnalysis
    PROGRAM = PunchoutProgram
    X_AXIS = SweepAxis.pulse("res_pulse", "freq")
    Y_AXIS = SweepAxis.pulse("res_pulse", "gain")
