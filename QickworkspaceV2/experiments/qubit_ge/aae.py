"""QubitGE power Rabi chevron experiment."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display, update_display
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

from ...core.base_experiment import BaseExperiment
from ...core.acquisition import acquire_values
from ...core.experiment_data import ExperimentData, QualityFlag
from ...core.base_program import BaseProgram


class PowerRabiChevronProgram(BaseProgram):
    """QICK program for power Rabi chevron: repeats the pulse ``iteration`` times."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_pulse")

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
        for _ in range(cfg["iteration"]):
            self.pulse(ch=cfg["qb_ch"], name="qb_pulse", t=0)
            self.delay_auto(t=0.02)
        self.delay_auto(t=0.05, tag="waiting")
        self.measure(cfg)


# Aliases for backward compatibility
PowerRabiProgram = PowerRabiChevronProgram


class PowerRabiChevron(BaseExperiment):
    """
    Power Rabi Chevron experiment.

    Performs a 2D scan: inner loop sweeps gain (hardware), outer loop
    sweeps iteration count (software).
    """

    EXPT_NAME = "s005_power_rabi_chevron"
    TAG = "PowerRabi"
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
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        self.cfg.setdefault("iteration", self.cfg.get("iter_start", 1))
        return PowerRabiChevronProgram(
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
        return prog.get_pulse_param("qb_pulse", "gain", as_array=True)

    def _extract_sweep_axis_y(self, prog):
        """Extract the secondary sweep axis from the program.

        Parameters
        ----------
        prog : Any
            Value for ``prog``.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self._sweep_vals_y

    def _build_scan_axes(self):
        """Build scan axes.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = self.cfg
        prog = self._create_program()
        gains = self._resolve_axis(
            self._extract_sweep_axis(prog), cfg.get("steps")
        )
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
        """Run the operation.

        Parameters
        ----------
        py_avg : Any
            Number of Python-level acquisition averages.
        show_final_plot : Any, default: False
            Whether to show final plot.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Result of the operation.
        """
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

        plot_display_id = f"live-plot-rabi-chevron-{np.random.randint(1e9)}"
        display(fig, display_id=plot_display_id)

        interrupted = False
        completed_rows = 0
        try:
            for y_idx, iter_val in enumerate(
                tqdm(iters, desc="Outer Sweep: Iterations")
            ):
                self.cfg["iteration"] = int(iter_val)
                prog = self._create_program()

                iq_data_row = acquire_values(
                    prog,
                    self.soc,
                    rounds=py_avg,
                    progress=False,
                )

                iqdata_full[y_idx, :] = iq_data_row
                completed_rows = y_idx + 1
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

        if interrupted and completed_rows:
            print(f"Interrupted after iteration {iters[completed_rows - 1]}.")

        self.iqdata = iqdata_full[:completed_rows]
        self._sweep_vals_y = iters[:completed_rows]
        if completed_rows == 0:
            result = ExperimentData(
                experiment_type=self.EXPT_NAME,
                quality=QualityFlag.BAD,
                quality_message="No data acquired",
                interrupted=True,
            )
            self.result = result
            return result

        optimal_gain = self._post_fit()
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=gains,
            y_axis=self._sweep_vals_y,
            fit_params=self.fit_params,
            fit_errors=self.fit_errors,
            fit_result={"optimal_gain": (optimal_gain, None)},
            scalar_result=float(optimal_gain),
            figures=[self._last_analysis_figure],
            quality=QualityFlag.NO_INFORMATION,
            interrupted=interrupted,
            avg_count=py_avg,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name=self.Y_SAVE_NAME,
            y_unit=self.Y_SAVE_UNIT,
            y_scale=self.Y_SAVE_SCALE,
            metadata={"iq_process": "abs"},
            dataset_dims={"iq": ["y", "x"]},
            analysis_data={
                "summed_signal": {
                    "values": np.sum(np.abs(self.iqdata), axis=0),
                    "dims": ["x"],
                }
            },
        )
        self.result = result
        return result

    def analyze_and_plot(self):
        """Return the analyze and plot result.

        Returns
        -------
        Any
            Result of the operation.
        """
        return self._post_fit()

    def _post_fit(self, x_vals=None):
        """Fit the acquired data after acquisition.

        Parameters
        ----------
        x_vals : Any, default: None
            Independent-variable values.

        Returns
        -------
        Any
            Result of the operation.
        """
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
            """Return the sinc2 model result.

            Parameters
            ----------
            x : Any
                Independent-variable values.
            A : Any
                Value for ``A``.
            x0 : Any
                Value for ``x0``.
            width : Any
                Value for ``width``.
            offset : Any
                Value for ``offset``.

            Returns
            -------
            Any
                Result of the operation.
            """
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
        self._last_analysis_figure = fig

        optimal_gain = float(optimal_gain)
        self.fit_params = np.array([optimal_gain])
        self.fit_errors = None
        self._chevron_fit_result = {"optimal_gain": optimal_gain}
        return optimal_gain

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
        if getattr(self, "_chevron_fit_result", None):
            g = self._chevron_fit_result["optimal_gain"]
            return f"Power Rabi Chevron\nOptimal gain = {g}\n{dict_val}"
        return f"{dict_val}"

__all__ = [
    "PowerRabiProgram",
    "PowerRabiChevronProgram",
    "PowerRabiChevron",
]
