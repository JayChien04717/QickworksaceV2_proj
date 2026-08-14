"""
Resonator/res_spec — s002: Resonator spectroscopy (ge).
"""

from __future__ import annotations

import numpy as np

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment, SweepAxis
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
        frequency, transmission = self._resolve_n_resonator_trace(freq_mhz, s21)

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

    def plot_n_resonators(
        self,
        count=None,
        *,
        freq_mhz=None,
        s21=None,
        fitted_freqs_mhz=None,
        y_mode="abs",
        ax=None,
        show=True,
        store_result=True,
        marker_kwargs=None,
        **detection_options,
    ):
        """Plot a one-tone trace with multiple resonator frequencies marked.

        The method can either reuse frequencies from a previous
        :meth:`fit_n_resonators` call, accept explicit fitted frequencies, or
        run a fresh multi-resonator fit when ``count`` is supplied.

        Parameters
        ----------
        count : int, optional
            Number of resonators to fit before plotting. If omitted, stored
            multi-resonator fit results from the latest run are used when
            available.
        freq_mhz : array_like, optional
            Strictly increasing frequency samples in megahertz. If omitted,
            ``self.result.x_axis`` is used.
        s21 : array_like, optional
            Complex transmission values corresponding to ``freq_mhz``. If
            omitted, ``self.result.raw_iq`` is used.
        fitted_freqs_mhz : array_like, optional
            Pre-computed resonator frequencies in megahertz. Supplying this
            skips fitting and only draws the markers.
        y_mode : {'abs', 'db', 'phase'}, optional
            Quantity to plot on the y-axis. The default is ``'abs'``.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If omitted, a new figure and axes are
            created.
        show : bool, optional
            Whether to call :func:`matplotlib.pyplot.show`. The default is
            ``True``.
        store_result : bool, optional
            Whether a fresh fit should be stored in the latest result. The
            default is ``True``.
        marker_kwargs : dict, optional
            Keyword arguments forwarded to ``ax.axvline`` for resonator
            markers.
        **detection_options
            Additional options forwarded to :meth:`fit_n_resonators` when a
            fresh fit is needed.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plot.
        ax : matplotlib.axes.Axes
            Axes containing the plot.
        resonator_freqs_mhz : numpy.ndarray
            Resonator frequencies shown on the plot, in megahertz.

        Raises
        ------
        RuntimeError
            If no fitted frequencies are available and ``count`` is omitted.
        ValueError
            If the trace or plotting options are invalid.

        Examples
        --------
        Fit and plot four resonators from the latest one-tone result::

            fig, ax, frequencies = experiment.plot_n_resonators(count=4)
        """
        import matplotlib.pyplot as plt

        frequency, transmission = self._resolve_n_resonator_trace(freq_mhz, s21)
        uses_current_result = freq_mhz is None and s21 is None

        if fitted_freqs_mhz is not None:
            fitted_mhz = np.asarray(fitted_freqs_mhz, dtype=float)
        elif count is not None:
            fitted_mhz, _ = self.fit_n_resonators(
                count,
                freq_mhz=freq_mhz,
                s21=s21,
                store_result=store_result,
                **detection_options,
            )
        elif uses_current_result:
            stored = self.result.metadata.get("multi_resonator_fit", {})
            fitted_mhz = np.asarray(stored.get("frequencies_mhz", []), dtype=float)
            if fitted_mhz.size == 0:
                raise RuntimeError(
                    "No stored multi-resonator fit is available. Supply count "
                    "or fitted_freqs_mhz."
                )
        else:
            raise RuntimeError(
                "Supply count or fitted_freqs_mhz when plotting explicit arrays."
            )

        if fitted_mhz.ndim != 1 or fitted_mhz.size == 0:
            raise ValueError("fitted_freqs_mhz must be a non-empty 1D array.")
        if not np.all(np.isfinite(fitted_mhz)):
            raise ValueError("fitted_freqs_mhz must contain only finite values.")

        y_mode = str(y_mode).lower()
        if y_mode == "abs":
            y_values = np.abs(transmission)
            ylabel = "|S21| (ADC Units)"
        elif y_mode == "db":
            floor = np.finfo(float).tiny
            y_values = 20 * np.log10(np.maximum(np.abs(transmission), floor))
            ylabel = "|S21| (dB)"
        elif y_mode == "phase":
            y_values = np.unwrap(np.angle(transmission))
            ylabel = "Phase (rad)"
        else:
            raise ValueError("y_mode must be one of 'abs', 'db', or 'phase'.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.plot(frequency, y_values, color="C0", label="S21")

        marker_style = {
            "color": "C3",
            "linestyle": "--",
            "linewidth": 1.2,
            "alpha": 0.8,
        }
        if marker_kwargs is not None:
            marker_style.update(marker_kwargs)

        for index, resonator_freq in enumerate(np.sort(fitted_mhz), start=1):
            ax.axvline(resonator_freq, **marker_style)
            nearest = int(np.argmin(np.abs(frequency - resonator_freq)))
            ax.scatter(
                [resonator_freq],
                [y_values[nearest]],
                color=marker_style.get("color", "C3"),
                s=28,
                zorder=3,
            )
            ax.annotate(
                f"R{index}\n{resonator_freq:.4f} MHz",
                xy=(resonator_freq, y_values[nearest]),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
                color=marker_style.get("color", "C3"),
            )

        title = "Multi-Resonator One-Tone Fit"
        if fitted_mhz.size:
            title += f" ({fitted_mhz.size} resonators)"
        ax.set_title(title)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax, np.sort(fitted_mhz)

    def _resolve_n_resonator_trace(self, freq_mhz=None, s21=None):
        """Return a one-dimensional one-tone trace for multi-resonator tools."""
        if (freq_mhz is None) != (s21 is None):
            raise ValueError("freq_mhz and s21 must be supplied together.")

        if freq_mhz is None:
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
            transmission = np.asarray(s21, dtype=complex).squeeze()

        if frequency.ndim != 1 or transmission.ndim != 1:
            raise ValueError("freq_mhz and s21 must both be one-dimensional.")

        return frequency, transmission

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
