"""
Mux two-tone qubit spectroscopy using a QICK frequency loop.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2, QickSweep1D

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData
from ._common import fit_quality, fit_snr, project_iq


class MuxTwoToneProgram(AveragerProgramV2):
    """Mux two-tone program: one QICK freqloop, mux readout on active channels."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        res_ch = cfg["res_ch"]
        ro_chs = list(cfg["active_ro_chs"])

        self.declare_gen(
            ch=res_ch,
            nqz=1,
            ro_ch=ro_chs[0],
            mixer_freq=cfg.get("mixer_freq", 0),
            mux_freqs=cfg["res_freqs"],
            mux_gains=cfg["res_gains"],
            mux_phases=cfg["res_phases"],
        )
        for ch, idx in zip(ro_chs, cfg["active_slots"]):
            self.declare_readout(
                ch=ch,
                length=cfg["ro_length"],
                freq=cfg["res_freqs"][idx],
                phase=cfg["ro_phases"][idx],
                gen_ch=res_ch,
            )

        self.add_loop("freqloop", cfg["steps"])

        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            qb_ch = cfg["qb_ch"][idx]
            self.declare_gen(
                ch=qb_ch,
                nqz=cfg["nqz_qb"][idx],
            )

        self.add_pulse(
            ch=res_ch,
            name="mux_readout",
            style="const",
            length=cfg["res_length"],
            mask=cfg["mask"],
        )

        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            qb_ch = cfg["qb_ch"][idx]
            pulse_name = f"{name}_pulse"
            pulse_type = cfg["pulse_type"][idx]
            freq = cfg["qb_freq_ge"][idx] + QickSweep1D(
                "freqloop", cfg["start_freq"], cfg["stop_freq"]
            )
            if pulse_type == "const":
                self.add_pulse(
                    ch=qb_ch,
                    name=pulse_name,
                    style="const",
                    length=cfg["qb_flat_top_length_ge"][idx],
                    freq=freq,
                    phase=cfg["qb_phase"][idx],
                    gain=cfg["qb_gain_ge"][idx],
                )
            elif pulse_type == "arb":
                env_name = f"{name}_envelope"
                self.add_gauss(
                    ch=qb_ch,
                    name=env_name,
                    sigma=cfg["sigma_ge"][idx],
                    length=5 * cfg["sigma_ge"][idx],
                    even_length=True,
                )
                self.add_pulse(
                    ch=qb_ch,
                    name=pulse_name,
                    style="arb",
                    envelope=env_name,
                    freq=freq,
                    phase=cfg["qb_phase"][idx],
                    gain=cfg["qb_gain_ge"][idx],
                )
            else:
                env_name = f"{name}_envelope"
                self.add_gauss(
                    ch=qb_ch,
                    name=env_name,
                    sigma=cfg["sigma_ge"][idx],
                    length=5 * cfg["sigma_ge"][idx],
                    even_length=True,
                )
                self.add_pulse(
                    ch=qb_ch,
                    name=pulse_name,
                    style="flat_top",
                    envelope=env_name,
                    length=cfg["qb_flat_top_length_ge"][idx],
                    freq=freq,
                    phase=cfg["qb_phase"][idx],
                    gain=cfg["qb_gain_ge"][idx],
                )

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][idx], name=f"{name}_pulse", t=0)
        self.delay_auto(0.05)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxTwoTone(BaseExperiment):
    """Mux two-tone spectroscopy with QickSweep1D support."""

    EXPT_NAME = "s003_mux_twotone_ge"
    TAG = "MuxTwoTone"
    X_LABEL = "Qubit frequency (MHz)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Mux TwoTone"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def __init__(self, config):
        """Initialize the MuxTwoTone instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.freq_axis = None
        self.freq_axes = None

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        return MuxTwoToneProgram(
            self.soccfg,
            reps=cfg["reps"],
            final_delay=cfg["relax_delay"],
            cfg=cfg,
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
        return self.freq_axis

    @staticmethod
    def _sweep_iq(iq_list, n_trace):
        """Return the sweep iq result.

        Parameters
        ----------
        iq_list : Any
            Value for ``iq_list``.
        n_trace : Any
            Value for ``n_trace``.

        Returns
        -------
        Any
            Result of the operation.
        """
        vals = []
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx][0]).squeeze()
            if arr.ndim > 0 and arr.shape[-1] == 2:
                vals.append(np.asarray(arr).dot([1, 1j]).reshape(-1))
            else:
                vals.append(np.asarray(arr, dtype=complex).reshape(-1))
        return np.asarray(vals, dtype=complex)

    def run(self, py_avg=1, span=50.0, iq_process="abs", plot=False):
        """Run the operation.

        Parameters
        ----------
        py_avg : Any, default: 1
            Number of Python-level acquisition averages.
        span : Any, default: 50.0
            Value for ``span``.
        iq_process : Any, default: 'abs'
            IQ processing mode.
        plot : Any, default: False
            Value for ``plot``.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        prog = MuxTwoToneProgram(
            self.soccfg,
            reps=cfg["reps"],
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )
        self._last_prog = prog

        qubit_names = list(cfg["qubit_names"])
        active_ro_chs = list(cfg["active_ro_chs"])
        active_slots = list(cfg["active_slots"])
        trace_count = len(active_ro_chs)
        self.freq_axes = {
            name: BaseExperiment._resolve_axis(
                prog.get_pulse_param(f"{name}_pulse", "freq", as_array=True),
                cfg["steps"],
            )
            for name in qubit_names
        }
        self.freq_axis = self.freq_axes[qubit_names[0]]

        interrupted = False
        try:
            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = self._sweep_iq(iq_list, trace_count)
        except KeyboardInterrupt:
            interrupted = True
            self.iqdata = None

        fit_result = {}
        fit_method = {}
        if self.iqdata is not None and np.isfinite(self.iqdata).any():
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            for idx, name in enumerate(qubit_names):
                trace = plot_data[idx]
                finite = np.isfinite(trace)
                if not np.any(finite):
                    continue
                freq_axis = self.freq_axes[name]
                freq, method = self._fit_qubit_freq(
                    freq_axis[finite], trace[finite]
                )
                fit_result[f"{name}_qb_freq_ge_mhz"] = (round(float(freq), 6), None)
                fit_method[name] = method

        figures = []
        if plot and self.iqdata is not None and np.isfinite(self.iqdata).any():
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 2.5 * trace_count)), squeeze=False
            )
            for ax, name, trace in zip(axes[:, 0], qubit_names, plot_data):
                freq_axis = self.freq_axes[name]
                ax.plot(freq_axis, trace, "o-", markersize=3)
                fit_freq = fit_result.get(f"{name}_qb_freq_ge_mhz", (None,))[0]
                if fit_freq is not None:
                    ax.axvline(fit_freq, c="r", ls="--", label=f"{fit_freq:.4f} MHz")
                    ax.legend()
                ax.set_ylabel(name)
                ax.set_xlabel(self.X_LABEL)
            axes[0, 0].set_title(
                self.TITLE_PREFIX + (" (Interrupted)" if interrupted else "")
            )
            fig.tight_layout()
            figures.append(fig)
            plt.show()

        has_data = self.iqdata is not None and np.isfinite(self.iqdata).any()
        model_fit_count = sum(method == "lorentzian" for method in fit_method.values())
        quality, quality_message = fit_quality(
            has_data, model_fit_count, trace_count, "Mux two-tone"
        )
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.freq_axis,
            fit_result=fit_result,
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "frequency_axes_mhz": {
                    name: np.asarray(self.freq_axes[name], dtype=float).tolist()
                    for name in qubit_names
                },
                "fit_method": fit_method,
                "span_mhz": float(span),
                "points_acquired": int(cfg["steps"]) if has_data else 0,
            },
            figures=figures,
            quality=quality,
            quality_message=quality_message,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result

    @staticmethod
    def _fit_qubit_freq(freq_axis_mhz, trace):
        """Fit qubit freq.

        Parameters
        ----------
        freq_axis_mhz : Any
            Value for ``freq_axis_mhz``.
        trace : Any
            Value for ``trace``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        try:
            from ..tools.fitting import fitlor, lorfunc

            popt, pcov, _ = fitlor(freq_axis_mhz, trace)
            f0 = float(popt[2])
            if (
                not np.isfinite(f0)
                or f0 < np.min(freq_axis_mhz)
                or f0 > np.max(freq_axis_mhz)
            ):
                raise RuntimeError("Lorentzian fit returned invalid f0")
            if fit_snr(trace, lorfunc(freq_axis_mhz, *popt)) < 3.0:
                raise RuntimeError("Lorentzian fit has insufficient SNR")
            return f0, "lorentzian"
        except Exception:
            min_idx = int(np.argmin(trace))
            max_idx = int(np.argmax(trace))
            median = np.median(trace)
            contrast_min = abs(trace[min_idx] - median)
            contrast_max = abs(trace[max_idx] - median)
            best_idx = min_idx if contrast_min >= contrast_max else max_idx
            return float(np.asarray(freq_axis_mhz)[best_idx]), "extremum"

    @staticmethod
    def _process_plot_data(iqdata, iq_process):
        """Prepare acquired data for plotting.

        Parameters
        ----------
        iqdata : Any
            Value for ``iqdata``.
        iq_process : Any
            IQ processing mode.

        Returns
        -------
        Any
            Result of the operation.
        """
        return project_iq(iqdata, iq_process)


__all__ = ["MuxTwoTone", "MuxTwoToneProgram"]
