"""
QubitGE/aae — s005a_AAE: Amplitude-Amplitude-Envelope (power Rabi chevron).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display, update_display
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...tools.fitting import (
    fit_probg_Xhalf,
    fit_probg_Xhalf_decay,
    probg_Xhalf,
    probg_Xhalf_decay,
)

AAE_GATE_GAP = 0.02


class PowerRabiChevronProgram(BaseProgram):
    """QICK program for AAE power Rabi: repeats the pulse ``iteration`` times."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        for _ in range(cfg["iteration"]):
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(t=0.02)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


# Aliases for backward compatibility
PowerRabiProgram = PowerRabiChevronProgram
AAEProgram = PowerRabiChevronProgram


class PowerRabiChevron(BaseExperiment):
    """
    AAE Power Rabi Chevron experiment.

    Performs a 2D scan: inner loop sweeps gain (hardware), outer loop
    sweeps iteration count (software).
    """

    EXPT_NAME = "s005_power_rabi_chevron"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ge"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ge"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0
    Y_SAVE_NAME = "Iterations"
    Y_SAVE_UNIT = "N"
    Y_SAVE_SCALE = 1.0

    def _create_program(self):
        self.cfg.setdefault("iteration", self.cfg.get("iter_start", 1))
        return PowerRabiChevronProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        return self._sweep_vals_y

    def _build_scan_axes(self):
        cfg = self.cfg
        prog = self._create_program()
        gains = self._extract_sweep_axis(prog)
        if "iter_start" in cfg and "iter_stop" in cfg:
            iters = np.arange(
                cfg["iter_start"],
                cfg["iter_stop"] + 1,
                cfg.get("iter_step", 1),
                dtype=int,
            )
        else:
            iters = np.array([int(cfg.get("iteration", 21))])
        return gains, iters

    def run(self, py_avg, show_final_plot=False, **kwargs):
        gains, iters = self._build_scan_axes()
        self._sweep_vals_x = gains
        self._sweep_vals_y = iters

        iqdata_full = np.zeros((len(iters), len(gains)), dtype=complex)
        data_to_plot = np.zeros((len(iters), len(gains)))

        fig, ax = plt.subplots(figsize=(6, 4))
        mesh = ax.pcolormesh(gains, iters, data_to_plot, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="ADC Units (Abs)")
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel("Iterations (N)")
        title = ax.set_title(f"{self.TITLE_PREFIX} (Initializing...)")

        plot_display_id = f"live-plot-aae-{np.random.randint(1e9)}"
        display(fig, display_id=plot_display_id)

        interrupted = False
        try:
            for y_idx, iter_val in enumerate(
                tqdm(iters, desc="Outer Sweep: Iterations")
            ):
                self.cfg["iteration"] = int(iter_val)
                prog = self._create_program()

                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                iq_data_row = iq_list[0][0].dot([1, 1j])

                iqdata_full[y_idx, :] = iq_data_row
                data_to_plot = np.abs(iqdata_full)

                mesh.set_array(data_to_plot.ravel())

                measured_data = data_to_plot[: y_idx + 1, :]
                c_min, c_max = np.min(measured_data), np.max(measured_data)
                if c_max > c_min:
                    mesh.set_clim(vmin=c_min, vmax=c_max)
                elif c_max > 0:
                    mesh.set_clim(vmin=0, vmax=c_max)

                title.set_text(f"{self.TITLE_PREFIX} | N={iter_val}")
                update_display(fig, display_id=plot_display_id)

        except KeyboardInterrupt:
            interrupted = True

        clear_output(wait=True)
        plt.close(fig)

        if interrupted:
            print(f"Interrupted at iteration {iters[y_idx]}.")

        self.iqdata = iqdata_full
        return self._post_fit()

    def analyze_and_plot(self):
        return self._post_fit()

    def _post_fit(self, x_vals=None):
        if self.iqdata is None:
            print("No data. Call run() first.")
            return None

        gains = self._sweep_vals_x
        iters = self._sweep_vals_y
        raw_sum_trace = np.sum(np.abs(self.iqdata), axis=0)
        sum_trace = gaussian_filter1d(raw_sum_trace, sigma=2.0)

        dx = gains[1] - gains[0]
        fft_vals = np.abs(np.fft.rfft(sum_trace - np.mean(sum_trace)))
        fft_freqs = np.fft.rfftfreq(len(gains), d=dx)
        freq_guess = fft_freqs[np.argmax(fft_vals[1:]) + 1]
        width_guess = 0.5 / freq_guess

        amp_guess = (np.max(sum_trace) - np.min(sum_trace)) / 2
        off_guess = np.mean(sum_trace)

        idx_max = int(np.argmax(sum_trace))
        idx_min = int(np.argmin(sum_trace))
        if abs(sum_trace[idx_max] - off_guess) >= abs(sum_trace[idx_min] - off_guess):
            x0_guess = gains[idx_max]
            sign_guess = 1.0
        else:
            x0_guess = gains[idx_min]
            sign_guess = -1.0

        def sinc2_model(x, A, x0, width, offset):
            return A * np.sinc((x - x0) / width) ** 2 + offset

        fit_success = False
        optimal_gain = x0_guess

        try:
            p0 = [sign_guess * amp_guess, x0_guess, width_guess, off_guess]
            popt, _ = curve_fit(
                sinc2_model,
                gains,
                sum_trace,
                p0=p0,
                bounds=(
                    [-np.inf, gains.min(), dx, -np.inf],
                    [np.inf, gains.max(), np.inf, np.inf],
                ),
                maxfev=10000,
            )
            A_fit, x0_fit, width_fit, offset_fit = popt
            if gains.min() <= x0_fit <= gains.max():
                optimal_gain = x0_fit
                fit_success = True
            else:
                print("Fit x0 out of range, falling back.")
        except Exception as e:
            print(f"Fit failed: {e}, falling back to smoothed extremum.")

        print(f"\n[PowerRabi] Optimal pi gain = {optimal_gain:.6f}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax0 = axes[0]
        im = ax0.pcolormesh(
            gains, iters, np.abs(self.iqdata), shading="auto", cmap="viridis"
        )
        fig.colorbar(im, ax=ax0, label="ADC Units (Abs)")
        ax0.axvline(
            optimal_gain,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=f"Fit={optimal_gain:.4f}",
        )
        ax0.set_title("Power Rabi Chevron")
        ax0.legend()

        ax1 = axes[1]
        ax1.scatter(
            gains, raw_sum_trace, s=20, color="steelblue", alpha=0.5, label="Raw Data"
        )
        ax1.plot(gains, sum_trace, "--", color="gray", alpha=0.7, label="Smoothed")

        if fit_success:
            fine_x = np.linspace(gains.min(), gains.max(), 2000)
            ax1.plot(
                fine_x,
                sinc2_model(fine_x, *popt),
                color="firebrick",
                lw=2,
                label="Sinc² Fit",
            )

        ax1.axvline(optimal_gain, color="red", linestyle="--")
        ax1.set_title("Summed Trace & Physical Fit")
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

        return optimal_gain

    def _save_comment(self, dict_val):
        if self.fit_params:
            g = self.fit_params.get("optimal_gain", "N/A")
            return f"AAE Power Rabi\nOptimal gain = {g}\n{dict_val}"
        return f"{dict_val}"


class GambettaFig2Program(BaseProgram):
    """QICK program for Gambetta Fig. 2: X90 followed by X90^n."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
        self.setup_standard_gates(cfg, prefix="ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.cooling_body(cfg)

        ch = cfg["qb_ch"]
        repetitions = int(cfg.get("aae_repetitions_current", 0))

        self.pulse(ch=ch, name="x90_ge", t=0)
        self.delay_auto(t=AAE_GATE_GAP)

        for _ in range(repetitions):
            self.pulse(ch=ch, name="x90_ge", t=0)
            self.delay_auto(t=AAE_GATE_GAP)

        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


class GambettaFig2AAE(BaseExperiment):
    """
    Gambetta/Sheldon Fig. 2 style pi/2 error-amplification calibration.

    Sequence: X90 followed by X90^n.  Sweeping n amplifies a small pi/2
    over/under-rotation into a measurable oscillation-frequency shift.
    """

    EXPT_NAME = "s005a_gambetta_fig2_aae_ge"
    TAG = "AAE"
    X_LABEL = "Additional X90 Pulses (n)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Gambetta Fig. 2 AAE ge"
    SWEEP_KEYS_TO_REMOVE = []
    X_SAVE_NAME = "Additional X90 Pulses"
    X_SAVE_UNIT = "N"
    X_SAVE_SCALE = 1.0

    def _normalize_cfg(self):
        if "pi2_gain_ge" not in self.cfg:
            self.cfg["pi2_gain_ge"] = self.cfg["pi_gain_ge"] / 2

    def _repetitions(self):
        reps = self.cfg.get("aae_repetitions")
        if reps is None:
            start = int(self.cfg.get("aae_repetitions_start", 0))
            stop = int(self.cfg.get("aae_repetitions_stop", 40))
            step = int(self.cfg.get("aae_repetitions_step", 1))
            reps = np.arange(start, stop + 1, step, dtype=int)
        return np.asarray(reps, dtype=int)

    def _create_program(self):
        self._normalize_cfg()
        return GambettaFig2Program(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return self._repetitions()

    def run(self, py_avg, iq_process="abs", show_final_plot=False, **kwargs):
        self.IQ_PROCESS = iq_process
        reps_axis = self._repetitions()
        self._sweep_vals_x = reps_axis
        self._sweep_vals_y = None

        iqdata = np.zeros(len(reps_axis), dtype=complex)
        data_to_plot = np.zeros(len(reps_axis), dtype=float)

        fig, ax = plt.subplots(figsize=(6, 4))
        (line,) = ax.plot(reps_axis, data_to_plot, marker="o", lw=1.5)
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel(self.Y_LABEL)
        title = ax.set_title(f"{self.TITLE_PREFIX} (Initializing...)")
        ax.grid(True, alpha=0.2)

        plot_display_id = f"live-plot-aae-fig2-{np.random.randint(1e9)}"
        display(fig, display_id=plot_display_id)

        interrupted = False
        last_idx = 0
        try:
            for idx, rep_val in enumerate(tqdm(reps_axis, desc="AAE repetitions")):
                last_idx = idx
                self.cfg["aae_repetitions_current"] = int(rep_val)
                prog = self._create_program()
                self._last_prog = prog

                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                arr = np.asarray(iq_list[0][0])
                if arr.ndim > 0 and arr.shape[-1] == 2:
                    iq_point = arr.dot([1, 1j]).reshape(-1)[0]
                else:
                    iq_point = arr.reshape(-1)[0]
                iqdata[idx] = iq_point

                if self.IQ_PROCESS == "real":
                    data_to_plot[idx] = np.real(iq_point)
                else:
                    data_to_plot[idx] = np.abs(iq_point)

                line.set_ydata(data_to_plot)
                measured = data_to_plot[: idx + 1]
                margin = 0.05 * max(np.ptp(measured), 1e-12)
                ax.set_ylim(np.min(measured) - margin, np.max(measured) + margin)
                title.set_text(f"{self.TITLE_PREFIX} | N={rep_val}")
                update_display(fig, display_id=plot_display_id)

        except KeyboardInterrupt:
            interrupted = True

        clear_output(wait=True)
        if not show_final_plot:
            plt.close(fig)

        if interrupted:
            print(f"Interrupted at N={reps_axis[last_idx]}.")
            reps_axis = reps_axis[: last_idx + 1]
            iqdata = iqdata[: last_idx + 1]

        self.iqdata = iqdata
        self._sweep_vals_x = reps_axis
        old_result = self._post_fit(reps_axis)

        from ...core.experiment_data import ExperimentData

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self._sweep_vals_x,
            y_axis=None,
            fit_params=self.fit_params,
            fit_errors=self.fit_errors,
            config=dict(self.cfg),
            interrupted=interrupted,
            avg_count=py_avg,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
        )
        if isinstance(old_result, dict):
            result.fit_result = self._build_fit_result()
        self.result = result
        return result

    def _processed_signal(self):
        if self.IQ_PROCESS == "real":
            signal = np.real(self.iqdata)
        elif self.IQ_PROCESS == "imag":
            signal = np.imag(self.iqdata)
        else:
            signal = np.abs(self.iqdata)
        return signal

    def _post_fit(self, x_vals=None):
        if self.iqdata is None:
            print("No data. Call run() first.")
            return None

        reps_axis = np.asarray(
            self._sweep_vals_x if x_vals is None else x_vals, dtype=float
        )
        y = np.asarray(self._processed_signal(), dtype=float)
        if len(reps_axis) < 5:
            print("Not enough points for AAE fit.")
            return None

        nominal_angle = np.pi / 2
        nominal_gain = float(self.cfg["pi2_gain_ge"])

        fit_success = False
        fit_model = "Xhalf_decay"
        popt, pcov, _ = fit_probg_Xhalf_decay(reps_axis, y)
        if np.any(np.isnan(popt)):
            fit_model = "Xhalf"
            popt, pcov, _ = fit_probg_Xhalf(reps_axis, y)
        if not np.any(np.isnan(popt)):
            fit_success = True

        if fit_model == "Xhalf_decay":
            offset, delta_deg, decay = popt
            model = probg_Xhalf_decay
            fit_decay = float(decay)
        else:
            offset, delta_deg = popt
            model = probg_Xhalf
            fit_decay = np.nan

        delta_err_deg = np.nan
        if pcov is not None and np.shape(pcov)[0] > 1:
            delta_err_deg = float(np.sqrt(np.abs(pcov[1, 1])))

        angle_error = float(delta_deg) * np.pi / 180
        angle_per_pulse = nominal_angle + angle_error
        corrected_gain = nominal_gain * nominal_angle / angle_per_pulse
        relative_gain_correction = corrected_gain / nominal_gain - 1.0

        self.fit_params = {
            "target": "pi2",
            "target_angle_rad": nominal_angle,
            "nominal_gain": nominal_gain,
            "angle_per_pulse_rad": float(angle_per_pulse),
            "angle_error_rad": float(angle_error),
            "angle_error_deg": float(delta_deg),
            "corrected_gain": float(corrected_gain),
            "relative_gain_correction": float(relative_gain_correction),
            "fit_offset": float(offset),
            "fit_delta_deg": float(delta_deg),
            "fit_decay": fit_decay,
            "fit_model": fit_model,
            "fit_success": fit_success,
        }
        self.fit_errors = {
            "angle_per_pulse_rad": float(delta_err_deg * np.pi / 180),
            "angle_error_rad": float(delta_err_deg * np.pi / 180),
            "angle_error_deg": delta_err_deg,
            "fit_delta_deg": delta_err_deg,
        }

        print(
            f"\n[AAE Fig.2] X90 angle error="
            f"{self.fit_params['angle_error_deg']:.4f} deg, "
            f"gain {nominal_gain:.6g} -> {corrected_gain:.6g}"
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(reps_axis, y, marker="o", lw=1.5, color="steelblue", label="Data")
        fit_y = model(reps_axis, *popt)
        ax.plot(reps_axis, fit_y, marker="s", ms=4, color="firebrick", lw=2, label=fit_model)
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel(self.Y_LABEL)
        ax.set_title(
            f"AAE Fig.2 X90: {self.fit_params['angle_error_deg']:.4f} deg/gate"
        )
        ax.grid(True, alpha=0.2)
        ax.legend()
        plt.tight_layout()
        plt.show()

        return dict(self.fit_params)

    def _build_fit_result(self):
        if not self.fit_params:
            return {}
        return {
            key: (val, self.fit_errors.get(key) if self.fit_errors else None)
            for key, val in self.fit_params.items()
        }

    def _save_comment(self, dict_val):
        if self.fit_params:
            return (
                "Gambetta Fig. 2 AAE\n"
                f"angle_error_deg={self.fit_params['angle_error_deg']:.6f}\n"
                f"corrected_gain={self.fit_params['corrected_gain']:.8g}\n"
                f"{dict_val}"
            )
        return f"{dict_val}"


# Alias
AAE = GambettaFig2AAE
AAEFig2 = GambettaFig2AAE

__all__ = [
    "PowerRabiProgram",
    "PowerRabiChevronProgram",
    "PowerRabiChevron",
    "AAEProgram",
    "AAE",
    "GambettaFig2Program",
    "GambettaFig2AAE",
    "AAEFig2",
]
