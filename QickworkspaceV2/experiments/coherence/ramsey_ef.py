"""
Ramsey EF (s012) — T2* and detuning for the ef transition.
"""

from __future__ import annotations

from ...analysis.qubit import RamseyEfAnalysis
from ...core.base_experiment import BaseExperiment, SweepAxis
from ...core.base_program import BaseProgram


class RamseyEfProgram(BaseProgram):
    """EF Ramsey: ge π → ef π/2 — wait — ef π/2."""

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
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse1", gain_key="pi2_gain_ef")
        ramsey_phase = (
            cfg.get("qb_phase_ef", 0) + cfg["wait_time"] * 360 * cfg["virtual_detune"]
        )
        self.setup_qb_pulse(
            cfg, "ef", name="qb_ef_pulse2", gain_key="pi2_gain_ef", phase=ramsey_phase
        )

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
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse1", t=0)
        self.delay_auto(cfg["wait_time"] + 0.01, tag="wait")
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse2", t=0)
        if cfg.get("ge_ref", False):
            self.delay_auto(0.02)
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class RamseyEf(BaseExperiment):
    """EF Ramsey (s012): extract T2* and detuning for ef transition."""

    EXPT_NAME = "s012_Ramsey_ef"
    TAG = "Ramsey"
    X_LABEL = "Delay time (us)"
    TITLE_PREFIX = "Qubit Ramsey ef"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Delay time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = RamseyEfAnalysis
    PROGRAM = RamseyEfProgram
    X_AXIS = SweepAxis.time("wait")

    def correct_detune(self):
        """Correct qubit ef frequency based on fitted detuning.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        if self.result is None:
            raise RuntimeError("Run the experiment first.")
        detune = self.result.fit_result.get("detune_MHz", (None,))[0]
        if detune is None:
            print("Detune not available (virtual_detune=0 or fit failed).")
            return self.cfg["qb_freq_ef"]
        if abs(detune - self.cfg["virtual_detune"]) > 0.005:
            self.cfg["qb_freq_ef"] = self.cfg["qb_freq_ef"] - round(
                (detune - self.cfg["virtual_detune"]), 2
            )
            print(f"over detune {round((detune - self.cfg['virtual_detune']), 5)}MHz")
            return round(self.cfg["qb_freq_ef"], 5)
        else:
            print("Detune < 5kHz")
            return self.cfg["qb_freq_ef"]

    def _save_comment(self, dict_val):
        """Return the comment stored with the result.

        Parameters
        ----------
        dict_val : Any
            Value for ``dict_val``.

        Returns
        -------
        Any
            Result of the operation.
        """
        if self.result is not None:
            T2 = self.result.fit_result.get("T2r_us", (None,))[0]
            if T2 is not None:
                return f"T2 Ramsey = {T2:.2f} us\n{dict_val}"
        return str(dict_val)
