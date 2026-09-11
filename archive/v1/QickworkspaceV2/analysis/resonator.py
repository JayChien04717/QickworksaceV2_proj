"""
Resonator analysis classes — circle fit, Lorentzian, hanger.
"""

from __future__ import annotations

import numpy as np

from ..core.base_analysis import BaseAnalysis
from ..core.experiment_data import ExperimentData, QualityFlag


class ResonatorSpecAnalysis(BaseAnalysis):
    """
    Analysis for resonator spectroscopy (s002).

    Applies circle fit (ABCD / hanger model) and extracts f0, Qi, Qc, Ql, κ.
    """

    thresholds = {
        "Qi": {"min": 1_000},
        "Ql": {"min": 500},
    }

    def _run(self, data: ExperimentData) -> None:
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.x_axis is None or data.raw_iq is None:
            return

        freqs = data.x_axis  # MHz
        iq = data.raw_iq

        try:
            try:
                from abcd_rf_fit import analyze
            except ImportError:
                from ..tools.abcd_rf_fit.abcd_rf_fit import analyze

            solve_type = data.config.get("_solve_type", "hm")
            fit = analyze(freqs * 1e6, iq, solve_type, fit_edelay=True)
            p = fit.tolist()
            f0, kappa, kappa_c = p[0], p[1], p[2]

            Qi = round(f0 / (kappa - kappa_c)) if kappa > kappa_c else 0
            Qc = round(f0 / kappa_c) if kappa_c > 0 else 0
            Ql = round(f0 / kappa) if kappa > 0 else 0

            data.fit_result = {
                "f_res[MHz]": (round(f0 / 1e6, 4), None),
                "Qi": (Qi, None),
                "Qc": (Qc, None),
                "Ql": (Ql, None),
                "kappa_MHz": (round(kappa * 1e-6, 4), None),
            }
            data.scalar_result = f0 / 1e6  # MHz
            data.metadata.update({"fit_model": f"abcd:{solve_type}", "fit_channel": "abs"})
            data.analysis_data.update({
                "fit_input": {"values": np.asarray(iq), "dims": ["x"]},
                "fit_curve": {"values": np.asarray(fit(freqs * 1e6)), "dims": ["x"]},
            })

        except Exception as exc:
            self._lorentzian_fallback(data, freqs, iq, exc)

    def plot(self, data: ExperimentData) -> None:
        """Render the stored ABCD fit with the framework's standard dashboard."""
        from ..plotter.plot_utils import plot_fit_result

        if data.x_axis is None or data.raw_iq is None:
            return

        f0 = (
            data.fit_result.get("f_res[MHz]")
            or data.fit_result.get("f0_MHz")
            or (None,)
        )[0]
        kappa = data.fit_result.get("kappa_MHz", (None,))[0]
        qi = data.fit_result.get("Qi", (None,))[0]
        qc = data.fit_result.get("Qc", (None,))[0]

        lines = []
        if f0 is not None:
            lines.append(f"f_res     = {f0:.3f} MHz")
        if kappa is not None:
            lines.append(f"linewidth = {kappa:.3f} MHz")
        if qi is not None:
            lines.append(f"Qi        = {int(qi):,}")
        if qc is not None:
            lines.append(f"Qc        = {int(qc):,}")

        fit_entry = data.analysis_data.get("fit_curve", {})
        fit_curve = (
            fit_entry.get("values")
            if isinstance(fit_entry, dict)
            else fit_entry
        )
        fit_params = None

        def fit_function(values, *_):
            return np.zeros_like(values, dtype=float)

        channel_curves = {}
        if fit_curve is not None:
            fit_curve = np.asarray(fit_curve).reshape(-1)
            x_values = np.asarray(data.x_axis, dtype=float).reshape(-1)
            if fit_curve.size == x_values.size:
                amplitude_curve = np.abs(fit_curve)

                def fit_function(values, *_):
                    return np.interp(values, x_values, amplitude_curve)

                fit_params = np.empty(0)
                if np.iscomplexobj(fit_curve):
                    channel_curves = {
                        "abs": amplitude_curve,
                        "real": np.real(fit_curve),
                        "imag": np.imag(fit_curve),
                        "phase": np.unwrap(np.angle(fit_curve)),
                    }
                else:
                    channel_curves = {"abs": amplitude_curve}

        quality = (
            data.quality.value
            if data.quality is not None
            else "no_information"
        )
        figure = plot_fit_result(
            np.asarray(data.x_axis),
            np.asarray(data.raw_iq),
            fit_function,
            fit_params,
            x_label="Frequency (MHz)",
            title="Resonator Spectroscopy",
            result_text="\n".join(lines),
            quality=quality,
            fit_channel="abs",
            channel_fit_curves=channel_curves,
        )
        if all(id(saved) != id(figure) for saved in data.figures):
            data.figures.append(figure)
        return figure

    def _lorentzian_fallback(self, data, freqs, iq, original_exc):
        """Fit a Lorentzian if circle fit fails.

        Parameters
        ----------
        data : Any
            Input data to process.
        freqs : Any
            Value for ``freqs``.
        iq : Any
            Value for ``iq``.
        original_exc : Any
            Value for ``original_exc``.
        """
        try:
            from ..tools.fitting import fitlor

            popt, pcov, _ = fitlor(freqs, np.abs(iq))
            err = np.sqrt(np.diag(pcov))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "f0_MHz": (popt[2], err[2]),
                "kappa_MHz": (2 * abs(popt[3]), 2 * err[3]),
            }
            data.scalar_result = popt[2]
            from ..tools.fitting import lorfunc

            fit_curve = lorfunc(freqs, *popt)
            data.metadata.update({"fit_model": "lorentzian", "fit_channel": "abs"})
            data.analysis_data.update({
                "fit_input": {"values": np.abs(iq), "dims": ["x"]},
                "fit_curve": {"values": np.asarray(fit_curve), "dims": ["x"]},
                "residual": {"values": np.abs(iq) - fit_curve, "dims": ["x"]},
            })
            data.quality_message = (
                f"circle fit failed ({original_exc}); used Lorentzian"
            )
        except Exception:
            data.quality = QualityFlag.BAD
            data.quality_message = f"all fits failed: {original_exc}"


class DispersiveShiftAnalysis(BaseAnalysis):
    """Empirical |g>/|e> resonator shift and readout-frequency SNR."""

    thresholds = {}

    def _run(self, data: ExperimentData) -> None:
        """Extract empirical g/e resonance points without any curve fit."""
        traces = np.asarray(data.raw_iq)
        frequency = np.asarray(data.x_axis, dtype=float)
        if frequency.ndim != 1 or traces.shape != (2, frequency.size):
            data.quality = QualityFlag.BAD
            data.quality_message = "Expected raw_iq with shape (2, frequency)"
            return

        responses = []
        resonance_indices = []
        for trace in traces:
            magnitude = np.abs(trace)
            # Hanger/notch response: smooth three bins, then take the empirical
            # minimum. This rejects single-bin noise without a resonator fit.
            padded = np.pad(magnitude, (1, 1), mode="edge")
            smoothed = np.convolve(padded, np.ones(3) / 3.0, mode="valid")
            responses.append(smoothed)
            resonance_indices.append(int(np.nanargmin(smoothed)))

        f_g = float(frequency[resonance_indices[0]])
        f_e = float(frequency[resonance_indices[1]])
        shift = f_e - f_g
        data.fit_result = {
            "f_res_g_MHz": (f_g, None),
            "f_res_e_MHz": (f_e, None),
            "resonator_shift_MHz": (shift, None),
            "abs_resonator_shift_MHz": (abs(shift), None),
            "chi_MHz": (shift / 2.0, None),
            "abs_chi_MHz": (abs(shift) / 2.0, None),
        }
        data.analysis_data["dispersive_response"] = {
            "values": np.asarray(responses),
            "dims": ["state", "frequency"],
        }
        data.metadata["chi_method"] = "empirical_smoothed_magnitude_minimum"
        data.scalar_result = shift / 2.0
        data.quality = QualityFlag.GOOD
    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        import matplotlib.pyplot as plt

        traces = np.asarray(data.raw_iq)
        if data.x_axis is None or traces.ndim < 2:
            return
        f_g = data.get_param("f_res_g_MHz")
        f_e = data.get_param("f_res_e_MHz")
        chi = data.get_param("chi_MHz")

        snr_entry = data.analysis_data.get("readout_snr")
        snr = snr_entry.get("values") if isinstance(snr_entry, dict) else snr_entry
        has_snr = snr is not None and np.size(snr) == np.size(data.x_axis)
        fig, axes = plt.subplots(
            2 if has_snr else 1, 1,
            figsize=(9, 8 if has_snr else 5), sharex=has_snr,
        )
        axes = np.atleast_1d(axes)
        ax = axes[0]
        ax.plot(data.x_axis, np.abs(traces[0]), label="|g>")
        ax.plot(data.x_axis, np.abs(traces[1]), label="|e>")
        if f_g is not None:
            ax.axvline(f_g, color="C0", linestyle="--", alpha=0.7)
        if f_e is not None:
            ax.axvline(f_e, color="C1", linestyle="--", alpha=0.7)
        title = "Dispersive Shift"
        if chi is not None:
            title += f" | chi = {chi:.4f} MHz, 2chi = {2 * chi:.4f} MHz"
        ax.set(title=title, ylabel="|IQ| (ADC unit)")
        ax.legend()
        ax.grid(alpha=0.25)

        if has_snr:
            snr = np.asarray(snr, dtype=float)
            best_frequency = data.get_param("best_readout_frequency_MHz")
            best_index = int(np.nanargmax(snr))
            if best_frequency is None:
                best_frequency = float(np.asarray(data.x_axis)[best_index])
            snr_ax = axes[1]
            snr_ax.plot(data.x_axis, snr, ".-", color="C2")
            snr_ax.axvline(
                best_frequency, color="red", linestyle="--",
                label=f"best = {best_frequency:.6f} MHz",
            )
            snr_ax.scatter(
                [best_frequency], [snr[best_index]],
                marker="*", s=120, color="red", zorder=3,
            )
            snr_ax.set(
                title=f"Readout-frequency SNR | max = {snr[best_index]:.3f}",
                xlabel="Frequency (MHz)", ylabel="SNR",
            )
            snr_ax.legend()
            snr_ax.grid(alpha=0.25)
        else:
            ax.set_xlabel("Frequency (MHz)")

        fig.tight_layout()
        if fig not in data.figures:
            data.figures.append(fig)
        plt.show()
        return fig


class ResonatorPunchoutAnalysis(BaseAnalysis):
    """Analysis for resonator punchout (s002b) — detect critical power."""

    thresholds = {}

    def _run(self, data: ExperimentData) -> None:
        # Punchout is primarily visual; store the 2D array summary
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.raw_iq is not None:
            data.fit_result = {"status": ("punchout_acquired", None)}


class LorentzianAnalysis(BaseAnalysis):
    """Generic Lorentzian analysis for qubit spectroscopy lines."""

    thresholds = {
        "linewidth_MHz": {"max": 100.0},
        "fit_channel_snr": {"min": 3.0},
    }

    def _run(self, data: ExperimentData) -> None:
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import fitlor, lorfunc

        try:
            _, popt, pcov, channel, score = self._fit_channel(data, fitlor, lorfunc)
            err = np.sqrt(np.diag(pcov))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "f0_MHz": (popt[2], err[2]),
                "linewidth_MHz": (2 * abs(popt[3]), 2 * err[3]),
                "amplitude": (popt[1], err[1]),
                "offset": (popt[0], err[0]),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            data.scalar_result = popt[2]
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"Lorentzian fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        from ..tools.fitting import lorfunc
        if data.fit_params is None:
            self._show_fit(
                data,
                lorfunc,
                None,
                xlabel="Frequency (MHz)",
                title="Qubit Spectroscopy | Fit unavailable",
                result_text=data.quality_message or "Lorentzian fit unavailable",
            )
            return
        f0 = data.fit_result.get("f0_MHz", (None,))[0]
        linewidth = data.fit_result.get("linewidth_MHz", (None,))[0]
        title = "Qubit Spectroscopy"
        if f0 is not None:
            title += f"  |  f0 = {f0:.3f} MHz"
        if linewidth is not None:
            title += f", linewidth = {linewidth:.3f} MHz"
        lines = []
        if f0 is not None:
            lines.append(f"f0        = {f0:.3f} MHz")
        if linewidth is not None:
            lines.append(f"linewidth = {linewidth:.3f} MHz")
        self._show_fit(
            data, lorfunc, data.fit_params,
            xlabel="Frequency (MHz)",
            title=title,
            result_text="\n".join(lines),
        )
