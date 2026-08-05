"""
Resonator/res_spec — s002: Resonator spectroscopy (ge).
"""

from __future__ import annotations

import numpy as np

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...analysis.resonator import ResonatorSpecAnalysis
from ...tools.fit_n_res import fit_n_resonators as _fit_n_resonators


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

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        return ResonatorSpecProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
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
        return prog.get_pulse_param("res_pulse", "freq", as_array=True)

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

    def fit_n_resonators(
        self,
        count,
        *,
        freq_mhz=None,
        s21=None,
        store_result=True,
        **detection_options,
    ):
        """Fit multiple resonator frequencies in a one-tone S21 trace.

        By default, the method operates on the most recent result from
        :meth:`run`. Explicit arrays may be supplied to analyze a different
        one-tone trace without running an acquisition first.

        Parameters
        ----------
        count : int
            Number of resonators to fit.
        freq_mhz : array_like, optional
            Strictly increasing frequency samples in megahertz. If omitted,
            ``self.result.x_axis`` is used.
        s21 : array_like, optional
            Complex transmission values corresponding to ``freq_mhz``. If
            omitted, ``self.result.raw_iq`` is used.
        store_result : bool, optional
            Whether to add the fitted frequencies and diagnostics to the most
            recent :class:`~QickworkspaceV2.core.experiment_data.ExperimentData`
            when fitting that result. The default is ``True``.
        **detection_options
            Additional options forwarded to
            :func:`QickworkspaceV2.tools.fit_n_res.detect_resonators`.

        Returns
        -------
        resonator_freqs_mhz : numpy.ndarray
            Sorted sub-bin resonator-frequency estimates in megahertz.
        fit_details : dict of str to numpy.ndarray
            Detection diagnostics. Frequency-valued fields produced by the
            lower-level fitter remain in hertz; ``fitted_freqs_mhz`` and
            ``sample_freqs_mhz`` are included for convenience.

        Raises
        ------
        RuntimeError
            If no completed one-tone result is available and explicit arrays
            were not supplied, or if too few resonator candidates are found.
        ValueError
            If only one explicit array is supplied or the trace is not
            one-dimensional.

        Examples
        --------
        Fit four resonators after acquiring a wide one-tone sweep::

            result = experiment.run(py_avg=10)
            frequencies, details = experiment.fit_n_resonators(count=4)
        """
        uses_current_result = freq_mhz is None and s21 is None
        if (freq_mhz is None) != (s21 is None):
            raise ValueError("freq_mhz and s21 must be supplied together.")

        if uses_current_result:
            if (
                self.result is None
                or self.result.x_axis is None
                or self.result.raw_iq is None
            ):
                raise RuntimeError(
                    "No completed one-tone result is available. Run the "
                    "experiment or supply freq_mhz and s21 explicitly."
                )
            frequency = np.asarray(self.result.x_axis, dtype=float)
            transmission = np.asarray(self.result.raw_iq, dtype=complex).squeeze()
        else:
            frequency = np.asarray(freq_mhz, dtype=float)
            transmission = np.asarray(s21, dtype=complex)

        if frequency.ndim != 1 or transmission.ndim != 1:
            raise ValueError("freq_mhz and s21 must both be one-dimensional.")

        fitted_hz, details = _fit_n_resonators(
            frequency * 1e6,
            transmission,
            count=int(count),
            **detection_options,
        )
        fitted_mhz = fitted_hz / 1e6
        details = dict(details)
        details["fitted_freqs_mhz"] = fitted_mhz
        details["sample_freqs_mhz"] = details["sample_freqs_hz"] / 1e6

        if uses_current_result and store_result:
            self.result.metadata["multi_resonator_fit"] = {
                "count": int(count),
                "frequencies_mhz": fitted_mhz.tolist(),
            }
            self.result.analysis_data["multi_resonator_indices"] = {
                "values": details["indices"],
                "dims": ["resonator"],
            }
            self.result.analysis_data["multi_resonator_freqs"] = {
                "values": fitted_mhz,
                "dims": ["resonator"],
                "unit": "MHz",
            }

        return fitted_mhz, details

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
