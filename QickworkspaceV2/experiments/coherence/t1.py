"""
Coherence/t1 — s008: T1 relaxation time (ge).
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.qubit import T1Analysis


class T1Program(BaseProgram):
    """QICK program for T1: π pulse then swept wait delay."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
        self.delay_auto(cfg["wait_time"] + 0.05, tag="wait")
        self.measure(cfg)


class T1(BaseExperiment):
    """
    T1 relaxation time (ge).

    Applies a π pulse and sweeps the wait delay before readout.
    Fits an exponential decay to extract T1.
    """

    EXPT_NAME = "s008_T1_ge"
    TAG = "T1"
    X_LABEL = "Times (us)"
    TITLE_PREFIX = "Qubit T1 ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Times"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = T1Analysis

    def _create_program(self):
        return T1Program(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        self.delay_times = prog.get_time_param("wait", "t", as_array=True)
        return self.delay_times

    def _save_comment(self, dict_val):
        if self.result is not None:
            T1 = self.result.fit_result.get("T1_us", (None,))[0]
            if T1 is not None:
                return f"T1 = {T1:.2f} us \n{dict_val}"
        return str(dict_val)
