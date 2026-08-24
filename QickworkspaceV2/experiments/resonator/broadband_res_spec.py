"""Broadband one-tone spectroscopy for locating several resonators."""

from __future__ import annotations

import numpy as np

from ...core.base_experiment import BaseExperiment, SweepAxis
from ...tools.fit_n_res import fit_n_resonators as _fit_n_resonators
from ...tools.fit_n_res import plot_n_resonators as _plot_n_resonators
from .res_spec import ResonatorSpec, ResonatorSpecProgram


class BroadbandResonatorSpec(ResonatorSpec):
    """Acquire a wide resonator sweep, then locate and plot every resonator.

    The hardware sequence is exactly :class:`ResonatorSpecProgram`.  This
    wrapper only replaces the single-resonator analysis with the existing
    multi-resonator fit and plot.
    """

    EXPT_NAME = "broadband_resonator_spectrum"
    INCLUDE_QUBIT_IN_FILENAME = False
    INCLUDE_QUBIT_IN_COMMENT = False
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Broadband Resonator Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = None
    PROGRAM = ResonatorSpecProgram
    X_AXIS = SweepAxis.pulse("res_pulse", "freq")

    def run(
        self,
        py_avg,
        count=4,
        *,
        y_mode="abs",
        show=True,
        detection_options=None,
        **kwargs,
    ):
        """Run the sweep and automatically fit/plot ``count`` resonators.

        ``solve_type`` is accepted in ``kwargs`` for compatibility with old
        ``ResonatorSpec`` notebook cells, but is not used by the broadband
        multi-resonator fit.
        """
        kwargs.pop("solve_type", None)
        result = BaseExperiment.run(self, py_avg, **kwargs)

        frequencies, _ = self.fit_n_resonators(
            count=count,
            **dict(detection_options or {}),
        )
        sweep_mhz, transmission = self._resolve_n_resonator_trace()
        fig, ax, _ = _plot_n_resonators(
            sweep_mhz * 1e6,
            transmission,
            frequencies * 1e6,
            y_mode=y_mode,
            show=show,
        )
        self.resonator_freqs = frequencies
        self.resonator_figure = fig
        self.resonator_axes = ax
        if all(existing is not fig for existing in result.figures):
            result.figures.append(fig)
        return result

    def fit_n_resonators(
        self,
        count,
        *,
        freq_mhz=None,
        s21=None,
        store_result=True,
        **detection_options,
    ):
        """Fit multiple resonator frequencies in a broadband S21 trace."""
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
        """Fit when needed and plot multiple broadband resonators."""
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

        fig, axes, fitted_hz = _plot_n_resonators(
            frequency * 1e6,
            transmission,
            fitted_mhz * 1e6,
            y_mode=y_mode,
            ax=ax,
            show=show,
            marker_kwargs=marker_kwargs,
        )
        return fig, axes, fitted_hz / 1e6

    def _resolve_n_resonator_trace(self, freq_mhz=None, s21=None):
        """Return a one-dimensional broadband one-tone trace."""
        if (freq_mhz is None) != (s21 is None):
            raise ValueError("freq_mhz and s21 must be supplied together.")

        if freq_mhz is None:
            if (
                self.result is None
                or self.result.x_axis is None
                or self.result.raw_iq is None
            ):
                raise RuntimeError(
                    "No completed broadband result is available. Run the "
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


__all__ = ["BroadbandResonatorSpec"]
