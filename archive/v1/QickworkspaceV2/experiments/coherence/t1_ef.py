"""
T1 EF (s013) — energy relaxation from |f⟩ to |e⟩.
"""

from __future__ import annotations

from ...analysis.qubit import T1Analysis
from ...core.base_experiment import BaseExperiment, SweepAxis
from ...core.base_program import BaseProgram


class T1EfProgram(BaseProgram):
    """EF T1: ge π → ef π → wait → readout."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pi", gain_key="pi_gain_ef")

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
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pi", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.01)
        self.delay_auto(0.02)
        self.measure(cfg)


class T1Ef(BaseExperiment):
    """EF T1 (s013): energy relaxation from |f⟩ to |e⟩."""

    EXPT_NAME = "s013_T1_ef"
    TAG = "T1"
    X_LABEL = "Delay time (us)"
    TITLE_PREFIX = "Qubit T1 ef"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Delay time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = T1Analysis
    PROGRAM = T1EfProgram
    X_AXIS = SweepAxis.time("wait")
