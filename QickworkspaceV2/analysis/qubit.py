"""
Qubit analysis classes — T1, Ramsey, SpinEcho, PowerRabi, etc.
"""

from __future__ import annotations

import numpy as np

from ..core.base_analysis import BaseAnalysis
from ..core.experiment_data import ExperimentData, QualityFlag



class T1Analysis(BaseAnalysis):
    """Exponential decay fit → T1 relaxation time."""

    thresholds = {
        "T1_us": {"min": 0.5, "max": 2000.0},
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
        from ..tools.fitting import expfunc, fitexp

        x = data.x_axis

        try:
            _, popt, pcov, channel, score = self._fit_channel(data, fitexp, expfunc)
            err = np.sqrt(np.diag(pcov))
            T1 = abs(float(popt[2]))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "T1_us": (T1, err[2]),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            data.scalar_result = T1
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"T1 fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.fit_params is None:
            return
        from ..tools.fitting import expfunc

        T1 = data.fit_result.get("T1_us", (None,))[0]
        lines = []
        if T1 is not None:
            lines.append(f"T1     = {T1:.2f} µs")
        self._show_fit(
            data,
            expfunc,
            data.fit_params,
            xlabel="Delay time (us)",
            title=f"T1 Relaxation  |  T1 = {T1:.2f} µs" if T1 else "T1 Relaxation",
            result_text="\n".join(lines),
        )




class RamseyAnalysis(BaseAnalysis):
    """Damped sinusoid fit → T2*, frequency detuning."""

    FREQUENCY_KEY = "qb_freq_ge"
    REQUIRED_CONFIG_KEYS = ("virtual_detune", FREQUENCY_KEY)

    thresholds = {
        "T2r_us": {"min": 0.1, "max": 1000.0},
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

        virtual_detune = self._config_value(data, "virtual_detune", 0.0)

        if virtual_detune != 0:
            self._fit_decaysin(data)
        else:
            self._fit_exp(data)

    def _fit_decaysin(self, data: ExperimentData) -> None:
        """Fit decaysin.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        from ..tools.fitting import decaysin, fitdecaysin

        x = data.x_axis

        try:
            _, popt, pcov, channel, score = self._fit_channel(
                data, fitdecaysin, decaysin
            )
            err = np.sqrt(np.diag(pcov))
            T2r = abs(float(popt[3]))
            detune = float(popt[1])
            virtual_detune = self._config_value(data, "virtual_detune", 0.0)
            qb_freq = self._config_value(data, self.FREQUENCY_KEY, 0.0)
            corrected_freq = qb_freq - round(detune - virtual_detune, 2)

            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "T2r_us": (T2r, err[3]),
                "detune_MHz": (detune, err[1]),
                "corrected_freq_MHz": (corrected_freq, None),
                "amplitude": (popt[0], err[0]),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            data.scalar_result = T2r
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"Ramsey decaysin fit failed: {exc}"

    def _fit_exp(self, data: ExperimentData) -> None:
        """Fit exp.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        from ..tools.fitting import expfunc, fitexp

        x = data.x_axis

        try:
            _, popt, pcov, channel, score = self._fit_channel(data, fitexp, expfunc)
            err = np.sqrt(np.diag(pcov))
            T2r = abs(float(popt[2]))
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "T2r_us": (T2r, err[2]),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            data.scalar_result = T2r
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"Ramsey exp fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.fit_params is None:
            return
        virtual_detune = self._config_value(data, "virtual_detune", 0.0)
        if virtual_detune != 0:
            from ..tools.fitting import decaysin as simfunc
        else:
            from ..tools.fitting import expfunc as simfunc
        T2r = data.fit_result.get("T2r_us", (None,))[0]
        detune = data.fit_result.get("detune_MHz", (None,))[0]
        corr = data.fit_result.get("corrected_freq_MHz", (None,))[0]
        title = "Ramsey"
        if T2r:
            title += f"  |  T2* = {T2r:.2f} µs"
        if detune:
            title += f",  Δf = {detune:.3f} MHz"
        lines = []
        if T2r:
            lines.append(f"T2*    = {T2r:.2f} µs")
        if detune:
            lines.append(f"detune = {detune:.3f} MHz")
        if corr:
            lines.append(f"f_corr = {corr:.3f} MHz")
        self._show_fit(
            data,
            simfunc,
            data.fit_params,
            xlabel="Delay time (us)",
            title=title,
            result_text="\n".join(lines),
        )




class SpinEchoAnalysis(BaseAnalysis):
    """Hahn echo fit -- T2E."""

    REQUIRED_CONFIG_KEYS = ("virtual_detune",)

    thresholds = {
        "T2e_us": {"min": 0.1, "max": 5000.0},
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

        virtual_detune = self._config_value(data, "virtual_detune", 0.0)
        x = data.x_axis
        detune = None
        detune_err = None

        try:
            if virtual_detune != 0:
                from ..tools.fitting import decaysin, fitdecaysin

                _, popt, pcov, channel, score = self._fit_channel(
                    data, fitdecaysin, decaysin
                )
                err = np.sqrt(np.diag(pcov))
                T2e = abs(float(popt[3]))
                T2e_err = err[3]
                detune = float(popt[1])
                detune_err = err[1]
            else:
                from ..tools.fitting import expfunc, fitexp

                _, popt, pcov, channel, score = self._fit_channel(data, fitexp, expfunc)
                err = np.sqrt(np.diag(pcov))
                T2e = abs(float(popt[2]))
                T2e_err = err[2]

            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "T2e_us": (T2e, T2e_err),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            if detune is not None:
                data.fit_result["detune_MHz"] = (detune, detune_err)
            data.scalar_result = T2e
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"SpinEcho fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.fit_params is None:
            return
        virtual_detune = self._config_value(data, "virtual_detune", 0.0)
        if virtual_detune != 0:
            from ..tools.fitting import decaysin as simfunc
        else:
            from ..tools.fitting import expfunc as simfunc

        T2e = data.fit_result.get("T2e_us", (None,))[0]
        detune = data.fit_result.get("detune_MHz", (None,))[0]

        title = "Spin Echo"
        lines = []
        if T2e:
            title += f"  |  T2E = {T2e:.2f} us"
            lines.append(f"T2E    = {T2e:.2f} us")
        if detune:
            title += f",  df = {detune:.3f} MHz"
            lines.append(f"detune = {detune:.3f} MHz")

        self._show_fit(
            data,
            simfunc,
            data.fit_params,
            xlabel="Delay time (us)",
            title=title,
            result_text="\n".join(lines),
        )


class RamseyEfAnalysis(RamseyAnalysis):
    """Ramsey analysis using the ef transition frequency for correction."""

    FREQUENCY_KEY = "qb_freq_ef"
    REQUIRED_CONFIG_KEYS = ("virtual_detune", FREQUENCY_KEY)


class PowerRabiAnalysis(BaseAnalysis):
    """Decaying sinusoid → π gain and π/2 gain."""

    thresholds = {
        "pi_gain": {"min": 0.05, "max": 0.95},
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
        from ..tools.fitting import fitsin, fix_phase, sinfunc

        x = data.x_axis

        try:
            _, popt, pcov, channel, score = self._fit_channel(data, fitsin, sinfunc)
            err = np.sqrt(np.diag(pcov))
            pi_gain, pi2_gain = fix_phase(popt)
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "pi_gain": (round(pi_gain, 6), None),
                "pi2_gain": (round(pi2_gain, 6), None),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            data.scalar_result = pi_gain
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"PowerRabi sine fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.fit_params is None:
            return
        from ..tools.fitting import sinfunc

        pi_gain = data.fit_result.get("pi_gain", (None,))[0]
        pi2_gain = data.fit_result.get("pi2_gain", (None,))[0]
        extra = []
        if pi_gain:
            extra.append(
                {
                    "x": pi_gain,
                    "color": "#d62728",
                    "ls": "--",
                    "lw": 1.2,
                    "label": f"π = {pi_gain:.4f}",
                }
            )
        if pi2_gain:
            extra.append(
                {
                    "x": pi2_gain,
                    "color": "#2ca02c",
                    "ls": "--",
                    "lw": 1.2,
                    "label": f"π/2 = {pi2_gain:.4f}",
                }
            )
        lines = []
        if pi_gain:
            lines.append(f"π     = {pi_gain:.4f}")
        if pi2_gain:
            lines.append(f"π/2   = {pi2_gain:.4f}")
        self._show_fit(
            data,
            sinfunc,
            data.fit_params,
            xlabel="Gain (a.u.)",
            title=f"Power Rabi  |  π gain = {pi_gain:.4f}" if pi_gain else "Power Rabi",
            result_text="\n".join(lines),
            extra_lines=extra,
        )




class TimeRabiAnalysis(BaseAnalysis):
    """Sinusoidal fit to time-domain Rabi → pi_length."""

    thresholds = {}

    def _run(self, data: ExperimentData) -> None:
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.x_axis is None or data.raw_iq is None:
            return
        from ..tools.fitting import decaysin, fitdecaysin

        x = data.x_axis

        try:
            _, popt, pcov, channel, score = self._fit_channel(
                data, fitdecaysin, decaysin
            )
            err = np.sqrt(np.diag(pcov))
            # pi time = 1 / (2 * frequency)
            pi_length = 1.0 / (2.0 * abs(popt[1])) if popt[1] != 0 else 0.0
            data.fit_params = np.array(popt)
            data.fit_errors = err
            data.fit_result = {
                "pi_length_us": (pi_length, None),
                "fit_channel": (channel, None),
                "fit_channel_snr": (score, None),
            }
            data.scalar_result = pi_length
        except Exception as exc:
            data.quality = QualityFlag.BAD
            data.quality_message = f"TimeRabi fit failed: {exc}"

    def plot(self, data: ExperimentData) -> None:
        """Plot the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if data.fit_params is None:
            return
        from ..tools.fitting import decaysin

        pi_len = data.fit_result.get("pi_length_us", (None,))[0]
        self._show_fit(
            data,
            decaysin,
            data.fit_params,
            xlabel="Pulse length (µs)",
            title=f"Time Rabi  |  π time = {pi_len:.3f} µs" if pi_len else "Time Rabi",
            result_text=f"π length = {pi_len:.3f} µs" if pi_len else "",
        )




class QubitTempAnalysis(BaseAnalysis):
    """Qubit temperature estimation from |e⟩ population."""

    thresholds = {
        "T_mK": {"max": 300.0},
    }

    def _run(self, data: ExperimentData) -> None:
        # Population ratio → temperature via Boltzmann
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        n_e = data.fit_result.get("n_excited", (None, None))[0]
        f_ge = data.config.get("qb_freq_ge", 5000.0)  # MHz

        if n_e is not None and 0 < n_e < 1:
            import scipy.constants as const

            f_hz = f_ge * 1e6
            ratio = n_e / (1 - n_e)
            if ratio > 0:
                T_K = (const.h * f_hz) / (const.k * np.log(1.0 / ratio))
                T_mK = T_K * 1000
                data.fit_result["T_mK"] = (round(T_mK, 1), None)
                data.scalar_result = T_mK




class SingleShotAnalysis(BaseAnalysis):
    """Single-shot readout quality metrics — fidelity, SNR, angle."""

    thresholds = {
        "fidelity": {"min": 0.85},
    }

    def _run(self, data: ExperimentData) -> None:
        # Actual analysis is handled inside SingleShot_gef.plot()
        # This class just wraps whatever the experiment already computed.
        """Run the operation.

        Parameters
        ----------
        data : ExperimentData
            Input data to process.
        """
        if "fidelity" not in data.fit_result:
            data.quality = QualityFlag.NO_INFORMATION
