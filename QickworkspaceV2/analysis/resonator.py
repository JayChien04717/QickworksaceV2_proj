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

        except Exception as exc:
            self._lorentzian_fallback(data, freqs, iq, exc)

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        import matplotlib.pyplot as plt

        if data.x_axis is None or data.raw_iq is None:
            return

        f0    = (data.fit_result.get("f_res[MHz]") or data.fit_result.get("f0_MHz") or (None,))[0]
        kappa = data.fit_result.get("kappa_MHz", (None,))[0]
        Qi    = data.fit_result.get("Qi",  (None,))[0]
        Qc    = data.fit_result.get("Qc",  (None,))[0]
        Ql    = data.fit_result.get("Ql",  (None,))[0]

        title = "Resonator Spectroscopy"
        if f0:    title += f"  |  f_res = {f0:.4f} MHz"
        if kappa: title += f",  κ = {kappa:.3f} MHz"

        # Lorentzian fallback (data.fit_params set) → use our standard panel
        if data.fit_params is None:
            try:
                try:
                    from abcd_rf_fit import analyze
                except ImportError:
                    from ..tools.abcd_rf_fit.abcd_rf_fit import analyze

                solve_type = data.config.get("_solve_type", "hm")
                fit_obj = analyze(
                    data.x_axis * 1e6, data.raw_iq, solve_type, fit_edelay=True
                )
                fit_obj.plot(title=title)
                plt.tight_layout()
                plt.show()
                return
            except Exception:
                pass  # fall through to Lorentzian panel

        from ..tools.fitting import lorfunc

        if data.fit_params is not None:
            fit_params = data.fit_params
        elif f0 is not None and kappa is not None:
            amp    = np.max(np.abs(data.raw_iq)) - np.min(np.abs(data.raw_iq))
            offset = np.min(np.abs(data.raw_iq))
            fit_params = np.array([offset, -amp, f0, kappa / 2])
        else:
            fit_params = None

        lines = []
        if f0:    lines.append(f"f_res  = {f0:.4f} MHz")
        if kappa: lines.append(f"κ      = {kappa:.3f} MHz")
        if Qi:    lines.append(f"Qi     = {int(Qi):,}")
        if Qc:    lines.append(f"Qc     = {int(Qc):,}")
        if Ql:    lines.append(f"Ql     = {int(Ql):,}")

        self._show_fit(
            data, lorfunc, fit_params,
            xlabel="Frequency (MHz)",
            title=title,
            result_text="\n".join(lines),
        )

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
                "kappa_MHz": (abs(popt[1]), err[1]),
            }
            data.scalar_result = popt[2]
            data.quality_message = (
                f"circle fit failed ({original_exc}); used Lorentzian"
            )
        except Exception:
            data.quality = QualityFlag.BAD
            data.quality_message = f"all fits failed: {original_exc}"


class DispersiveShiftAnalysis(BaseAnalysis):
    """Fit |g> and |e> resonator traces and report both 2chi and chi."""

    thresholds = {}

    def _run(self, data: ExperimentData) -> None:
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        traces = np.asarray(data.raw_iq)
        if data.x_axis is None or traces.ndim < 2 or traces.shape[0] != 2:
            data.quality = QualityFlag.BAD
            data.quality_message = "Expected raw_iq with shape (2, frequency)"
            return

        fitted = []
        for state, trace in zip(("g", "e"), traces):
            spectrum = ExperimentData(
                experiment_type=f"resonator_spec_{state}",
                raw_iq=np.squeeze(trace),
                x_axis=np.asarray(data.x_axis),
                config=dict(data.config),
            )
            ResonatorSpecAnalysis().run(spectrum)
            frequency = (
                spectrum.fit_result.get("f_res[MHz]")
                or spectrum.fit_result.get("f0_MHz")
            )
            if frequency is None:
                data.quality = QualityFlag.BAD
                data.quality_message = f"Could not fit resonator spectrum for |{state}>"
                return
            fitted.append((float(frequency[0]), frequency[1]))

        f_g, f_e = fitted[0][0], fitted[1][0]
        shift = f_e - f_g
        errors = [item[1] for item in fitted]
        shift_error = None
        if all(error is not None and np.isfinite(error) for error in errors):
            shift_error = float(np.hypot(*errors))
        data.fit_result = {
            "f_res_g_MHz": fitted[0],
            "f_res_e_MHz": fitted[1],
            "resonator_shift_MHz": (shift, shift_error),
            "abs_resonator_shift_MHz": (abs(shift), shift_error),
            "chi_MHz": (shift / 2, None if shift_error is None else shift_error / 2),
            "abs_chi_MHz": (
                abs(shift) / 2,
                None if shift_error is None else shift_error / 2,
            ),
        }
        data.scalar_result = shift / 2

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

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data.x_axis, np.abs(traces[0]), label="|g>")
        ax.plot(data.x_axis, np.abs(traces[1]), label="|e>")
        if f_g is not None:
            ax.axvline(f_g, color="C0", linestyle="--", alpha=0.7)
        if f_e is not None:
            ax.axvline(f_e, color="C1", linestyle="--", alpha=0.7)
        title = "Dispersive Shift"
        if chi is not None:
            title += f" | chi = {chi:.4f} MHz, 2chi = {2 * chi:.4f} MHz"
        ax.set(title=title, xlabel="Frequency (MHz)", ylabel="|IQ| (ADC unit)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()


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

        x = data.x_axis

        try:
            _, popt, pcov, channel, score = self._fit_channel(data, fitlor, lorfunc)
            err = np.sqrt(np.diag(pcov))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "f0_MHz": (popt[2], err[2]),
                "linewidth_MHz": (abs(popt[1]), err[1]),
                "amplitude": (popt[0], err[0]),
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
        if data.fit_params is None:
            return
        from ..tools.fitting import lorfunc
        f0    = data.fit_result.get("f0_MHz",        (None,))[0]
        kappa = data.fit_result.get("linewidth_MHz",  (None,))[0]
        title = "Qubit Spectroscopy"
        if f0:    title += f"  |  f0 = {f0:.3f} MHz"
        if kappa: title += f",  κ = {kappa:.3f} MHz"
        lines = []
        if f0:    lines.append(f"f0     = {f0:.3f} MHz")
        if kappa: lines.append(f"κ      = {kappa:.3f} MHz")
        self._show_fit(
            data, lorfunc, data.fit_params,
            xlabel="Frequency (MHz)",
            title=title,
            result_text="\n".join(lines),
        )
