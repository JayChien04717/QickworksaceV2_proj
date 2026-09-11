"""
Resonator/res_spec — s002: Resonator spectroscopy (ge).
"""

from __future__ import annotations

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment, SweepAxis
from ...analysis.resonator import ResonatorSpecAnalysis


class ResonatorSpecProgram(BaseProgram):
    """QICK program for resonator spectroscopy: sweeps resonator frequency."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg, prefix="ge")
        self.add_loop("freqloop", cfg["steps"])

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
        self.measure(cfg)


class ResonatorSpec(BaseExperiment):
    """
    Resonator spectroscopy (ge).

    Sweeps ``res_freq_ge`` and fits a circle (ABCD / hanger model) or
    Lorentzian to extract resonator parameters.
    """

    EXPT_NAME = "s002_res_ge"
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Resonator Spectroscopy"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = ResonatorSpecAnalysis
    PROGRAM = ResonatorSpecProgram
    X_AXIS = SweepAxis.pulse("res_pulse", "freq")

    def run(self, py_avg, solve_type="hm", **kwargs):
        """Run resonator spectroscopy.  ``solve_type`` passed to circle fit.

        Parameters
        ----------
        py_avg : Any
            Number of Python-level acquisition averages.
        solve_type : Any, default: 'hm'
            Value for ``solve_type``.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Result of the operation.
        """
        self.cfg["_solve_type"] = solve_type
        return super().run(py_avg, **kwargs)

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
            f0 = self.result.fit_result.get("f0_GHz", (None,))[0]
            if f0 is not None:
                return f"f_res = {f0 * 1000:.4f} MHz, \n{dict_val}"
        return str(dict_val)
