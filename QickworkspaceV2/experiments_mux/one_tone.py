"""
Mux one-tone resonator spectroscopy using a mux generator.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display, update_display
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag


class MuxOneToneProgram(AveragerProgramV2):
    """Single-frequency mux resonator spectroscopy point."""

    def _initialize(self, cfg):
        """Initialize pulse and acquisition resources.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        res_ch = cfg["res_ch"]
        ro_chs = list(cfg.get("active_ro_chs", cfg.get("ro_chs", [])))
        if not ro_chs:
            raise ValueError("MuxOneTone requires 'active_ro_chs' or 'ro_chs'.")

        self.declare_gen(
            ch=res_ch,
            nqz=1,
            ro_ch=ro_chs[0],
            mixer_freq=cfg.get("mixer_freq", 0),
            mux_freqs=cfg["res_freqs"],
            mux_gains=cfg["res_gains"],
            mux_phases=cfg["res_phases"],
        )

        for ch, freq, phase in zip(
            ro_chs, cfg["active_res_freqs"], cfg["active_ro_phases"]
        ):
            self.declare_readout(
                ch=ch,
                length=cfg["ro_length"],
                freq=freq,
                phase=phase,
                gen_ch=res_ch,
            )

        self.add_pulse(
            ch=res_ch,
            name="mux_readout",
            style="const",
            length=cfg["res_length"],
            mask=cfg["mask"],
        )

    def _body(self, cfg):
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])


class MuxOneTone(BaseExperiment):
    """Mux one-tone spectroscopy with one trace per active qubit/readout."""

    EXPT_NAME = "s002_mux_res_ge"
    TAG = "MuxOneTone"
    X_LABEL = "Detuning from center (MHz)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Mux OneTone"
    X_SAVE_NAME = "Detuning"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    def __init__(self, config):
        """Initialize the MuxOneTone instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.freq_offsets = None
        self.freq_axes = None

    def _normalized_cfg(self, *, freq_offset=0.0):
        """Normalize d cfg.

        Parameters
        ----------
        freq_offset : Any, default: 0.0
            Value for ``freq_offset``.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        cfg.setdefault("active_ro_chs", cfg.get("ro_chs", []))
        cfg.setdefault("active_slots", list(range(len(cfg.get("active_ro_chs", [])))))
        cfg.setdefault("mask", list(range(len(cfg.get("res_freqs", [])))))
        cfg.setdefault("res_phases", [0] * len(cfg["mask"]))
        cfg.setdefault("ro_phases", [0] * len(cfg["mask"]))
        cfg.setdefault("res_gains", [1] * len(cfg["mask"]))

        cfg["res_freqs"] = [float(f) + freq_offset for f in cfg["res_freqs"]]
        active_slots = list(cfg["active_slots"])
        cfg["active_res_freqs"] = [cfg["res_freqs"][slot] for slot in active_slots]
        cfg["active_ro_phases"] = [cfg["ro_phases"][slot] for slot in active_slots]
        return cfg

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        return MuxOneToneProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self._normalized_cfg(),
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
        return self.freq_offsets

    def _configured_offsets(self):
        """Return the configured offsets result.

        Returns
        -------
        Any
            Result of the operation.
        """
        sweep = self.cfg.get("res_freq_ge") if hasattr(self.cfg, "get") else None
        if sweep is None:
            return None

        arr = np.asarray(sweep)
        if arr.ndim != 1 or arr.size <= 1:
            return None

        slot_count = len(self.cfg.get("res_freqs", []))
        if arr.size == slot_count:
            return None

        try:
            return arr.astype(float)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _point_iq(iq_list, n_trace):
        """Return the point iq result.

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
                vals.append(np.asarray(arr).dot([1, 1j]).reshape(-1)[0])
            else:
                vals.append(np.asarray(arr, dtype=complex).reshape(-1)[0])
        return np.asarray(vals, dtype=complex)

    def run(self, py_avg=1, span=20.0, steps=101, iq_process="abs", plot=False):
        """Run the operation.

        Parameters
        ----------
        py_avg : Any, default: 1
            Number of Python-level acquisition averages.
        span : Any, default: 20.0
            Value for ``span``.
        steps : Any, default: 101
            Value for ``steps``.
        iq_process : Any, default: 'abs'
            IQ processing mode.
        plot : Any, default: False
            Value for ``plot``.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg0 = self._normalized_cfg()
        configured_offsets = self._configured_offsets()
        if configured_offsets is not None:
            self.freq_offsets = configured_offsets
        else:
            half_span = abs(float(span)) / 2.0
            self.freq_offsets = np.linspace(-half_span, half_span, int(steps))
        half_span = max(abs(float(np.min(self.freq_offsets))), abs(float(np.max(self.freq_offsets))))

        active_slots = list(cfg0["active_slots"])
        active_ro_chs = list(cfg0["active_ro_chs"])
        qubit_names = list(cfg0.get("qubit_names", [f"ro{ch}" for ch in active_ro_chs]))
        centers_if = np.asarray([cfg0["res_freqs"][slot] for slot in active_slots])
        lo_ext = cfg0.get("LO_ext") or 0
        centers_abs = centers_if + float(lo_ext)
        self.freq_axes = {
            name: (center + self.freq_offsets).tolist()
            for name, center in zip(qubit_names, centers_abs)
        }

        trace_count = len(active_ro_chs)
        self.iqdata = np.full(
            (trace_count, len(self.freq_offsets)), np.nan + 1j * np.nan, dtype=complex
        )
        point_done = 0
        interrupted = False

        if plot:
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 2.5 * trace_count)), squeeze=False
            )
            axes = axes[:, 0]
            lines = []
            for ax, name, center in zip(axes, qubit_names, centers_abs):
                (line,) = ax.plot(
                    center + self.freq_offsets,
                    np.full_like(self.freq_offsets, np.nan, dtype=float),
                    "o-",
                    markersize=3,
                    alpha=0.85,
                )
                ax.set_ylabel(name)
                ax.set_xlim(center - half_span, center + half_span)
                lines.append(line)
            axes[-1].set_xlabel("Frequency (MHz)")
            title = axes[0].set_title(f"{self.TITLE_PREFIX} | Point: 0 / 0")
            fig.tight_layout()
            plot_id = f"live-plot-mux-onetone-{np.random.randint(int(1e9))}"
            display(fig, display_id=plot_id)
        else:
            fig = axes = lines = title = plot_id = None

        try:
            for f_idx, offset in enumerate(tqdm(self.freq_offsets, desc="Frequency")):
                point_cfg = self._normalized_cfg(freq_offset=float(offset))
                prog = MuxOneToneProgram(
                    self.soccfg,
                    reps=point_cfg["reps"],
                    final_delay=point_cfg["relax_delay"],
                    cfg=point_cfg,
                )
                self._last_prog = prog
                iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                self.iqdata[:, f_idx] = self._point_iq(iq_list, trace_count)
                point_done = f_idx + 1

                if plot:
                    plot_data = self._process_plot_data(self.iqdata, iq_process)
                    for ax, line, trace in zip(axes, lines, plot_data):
                        line.set_ydata(trace)
                        finite = trace[np.isfinite(trace)]
                        if finite.size == 0:
                            continue
                        cmin, cmax = np.min(finite), np.max(finite)
                        span_y = max(cmax - cmin, 1e-9)
                        ax.set_ylim(cmin - 0.1 * span_y, cmax + 0.1 * span_y)
                    title.set_text(
                        f"{self.TITLE_PREFIX} | Point: {point_done} / {len(self.freq_offsets)}"
                    )
                    update_display(fig, display_id=plot_id)
        except KeyboardInterrupt:
            interrupted = True

        if plot:
            clear_output(wait=True)
            if fig is not None:
                plt.close(fig)

        figures = []
        fit_result = {}
        fit_method = {}
        if self.iqdata is not None and np.isfinite(self.iqdata).any():
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            for name, center, trace, iq_trace in zip(
                qubit_names, centers_abs, plot_data, self.iqdata
            ):
                finite = np.isfinite(trace) & np.isfinite(iq_trace)
                if not np.any(finite):
                    continue
                freq_axis = center + self.freq_offsets
                res_freq, method = MuxOneTone._fit_res_freq_mag_min(
                    freq_axis[finite], trace[finite]
                )
                fit_result[f"{name}_res_freq_mhz"] = (round(float(res_freq), 6), None)
                fit_method[name] = method

            if plot:
                final_fig, final_axes = plt.subplots(
                    trace_count,
                    1,
                    figsize=(8, max(3, 2.5 * trace_count)),
                    squeeze=False,
                )
                final_axes = final_axes[:, 0]
                for ax, name, center, trace in zip(
                    final_axes, qubit_names, centers_abs, plot_data
                ):
                    x = center + self.freq_offsets
                    ax.plot(x, trace, "o-", markersize=3)
                    res_freq = fit_result[f"{name}_res_freq_mhz"][0]
                    ax.axvline(res_freq, c="r", ls="--", label=f"{res_freq:.4f} MHz")
                    ax.set_ylabel(name)
                    ax.set_xlim(center - half_span, center + half_span)
                    ax.legend()
                final_axes[-1].set_xlabel("Frequency (MHz)")
                final_axes[0].set_title(
                    self.TITLE_PREFIX + (" (Interrupted)" if interrupted else "")
                )
                final_fig.tight_layout()
                display(final_fig)
                figures.append(final_fig)
                plt.close(final_fig)

        has_data = self.iqdata is not None and np.isfinite(self.iqdata).any()
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.freq_offsets,
            fit_result=fit_result,
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "center_freqs_mhz": centers_abs.tolist(),
                "frequency_axes_mhz": self.freq_axes,
                "fit_method": fit_method,
                "span_mhz": float(span),
                "points_acquired": point_done,
                "LO_ext": cfg0.get("LO_ext"),
            },
            figures=figures,
            quality=QualityFlag.GOOD if has_data else QualityFlag.BAD,
            quality_message="Mux one-tone acquired."
            if has_data
            else "No data acquired.",
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result

    @staticmethod
    def _fit_res_freq_mag_min(freq_axis_mhz, plot_trace):
        """Fit res freq mag min.

        Parameters
        ----------
        freq_axis_mhz : Any
            Value for ``freq_axis_mhz``.
        plot_trace : Any
            Value for ``plot_trace``.

        Returns
        -------
        Any
            Result of the operation.
        """
        idx = int(np.nanargmin(plot_trace))
        return float(np.asarray(freq_axis_mhz, dtype=float)[idx]), "mag_min"

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
        iq_process = (iq_process or "abs").lower()
        if iq_process in {"real", "i", "avgi"}:
            return np.real(iqdata)
        if iq_process in {"imag", "q", "avgq"}:
            return np.imag(iqdata)
        if iq_process == "phase":
            return np.unwrap(np.angle(iqdata), axis=-1)
        return np.abs(iqdata)


__all__ = ["MuxOneTone", "MuxOneToneProgram"]
