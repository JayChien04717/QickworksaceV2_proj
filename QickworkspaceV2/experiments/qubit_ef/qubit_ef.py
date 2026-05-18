"""
EF qubit spectroscopy experiments.
"""

from __future__ import annotations

from ...analysis.resonator import LorentzianAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram


class QubitSpecEfProgram(BaseProgram):
    """EF spectroscopy: ge pi pulse then sweep ef drive frequency."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.02)
        self.delay_auto(0.02)
        self.measure(cfg)


class QubitSpecEf(BaseExperiment):
    """Qubit spectroscopy (ef): ge pi then sweep ef frequency."""

    EXPT_NAME = "s010_qubit_spec_ef"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ef Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ef"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = LorentzianAnalysis

    def _create_program(self):
        return QubitSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_pulse", "freq", as_array=True)


__all__ = ["QubitSpecEfProgram", "QubitSpecEf"]
