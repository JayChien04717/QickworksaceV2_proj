"""
Coherence/spin_echo — s007: Spin Echo (Hahn echo, ge).
"""

from __future__ import annotations

from ...analysis.qubit import SpinEchoAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram


class SpinEchoProgram(BaseProgram):
    """QICK program for Hahn echo: π/2 — wait/2 — π — wait/2 — π/2."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("waitloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse1", gain_key="pi2_gain_ge")
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse_pi", gain_key="pi_gain_ge")
        ramsey_phase = (
            cfg.get("qb_phase", 0) + cfg["wait_time"] * 360 * cfg["virtual_detune"]
        )
        self.setup_qb_pulse(
            cfg, "ge", name="qb_pulse2", gain_key="pi2_gain_ge", phase=ramsey_phase
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
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse1", t=0)
        self.delay_auto((cfg["wait_time"] / 2) + 0.01, tag="wait1")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse_pi", t=0)
        self.delay_auto((cfg["wait_time"] / 2) + 0.01, tag="wait2")
        self.pulse(ch=cfg["qb_ch"], name="qb_pulse2", t=0)
        self.delay_auto(0.01)
        self.measure(cfg)


class SpinEcho(BaseExperiment):
    """Spin Echo (ge): Hahn echo, extracts T2 Echo."""

    EXPT_NAME = "s007_SpinEcho_ge"
    TAG = "Spin Echo"
    X_LABEL = "Delay time (us)"
    TITLE_PREFIX = "Qubit Spin Echo ge"
    SWEEP_KEYS_TO_REMOVE = ["wait_time"]
    X_SAVE_NAME = "Delay time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    Analysis = SpinEchoAnalysis

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        return SpinEchoProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Extract the primary sweep axis from the program.

        Parameters
        ----------
        prog : Any
            Value for ``prog``.

        Returns
        -------
        Any
            Result of the operation.
        """
        self.delay_times = prog.get_time_param(
            "wait1", "t", as_array=True
        ) + prog.get_time_param("wait2", "t", as_array=True)
        return self.delay_times

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
            T2 = self.result.fit_result.get("T2e_us", (None,))[0]
            if T2 is not None:
                return f"T2 Spin Echo = {T2:.2f} us\n{dict_val}"
        return str(dict_val)
