"""
Mux time-of-flight measurement using a mux generator and PFB readout channels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display, update_display
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag
from ..tools.system_tool import clean_config


class MuxTOFProgram(AveragerProgramV2):
    """QICK program for mux TOF acquire_decimated measurements."""

    def _initialize(self, cfg):
        res_ch = cfg["res_ch"]
        ro_chs = cfg.get("active_ro_chs", cfg.get("ro_chs"))
        active_slots = cfg.get("active_slots", list(range(len(ro_chs))))
        if not ro_chs:
            raise ValueError("MuxTOF requires 'active_ro_chs' or 'ro_chs'.")

        self.declare_gen(
            ch=res_ch,
            nqz=cfg.get("nqz_res", 1),
            ro_ch=ro_chs[0],
            mixer_freq=cfg.get("mixer_freq", 0),
            mux_freqs=cfg["res_freqs"],
            mux_gains=cfg["res_gains"],
            mux_phases=cfg["res_phases"],
        )

        for ch, slot in zip(ro_chs, active_slots):
            self.declare_readout(
                ch=ch,
                length=cfg["ro_len"],
                freq=cfg["res_freqs"][slot],
                phase=cfg["ro_phases"][slot],
                gen_ch=res_ch,
            )

        self.add_pulse(
            ch=res_ch,
            name="mux_readout",
            style="const",
            length=cfg["res_len"],
            mask=cfg["mask"],
        )

    def _body(self, cfg):
        trigger_chs = cfg.get("trigger_ro_chs", cfg.get("active_ro_chs", cfg["ro_chs"]))
        self.trigger(ros=trigger_chs, pins=[0], t=cfg["trig_time"], ddr4=False)
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxTOF(BaseExperiment):
    """Mux time-of-flight measurement with one trace per active qubit/readout."""

    EXPT_NAME = "s001_mux_tof"
    TAG = "MuxTOF"
    X_LABEL = r"Time ($\mu$s)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Mux Time of Flight"
    X_SAVE_NAME = "Time"
    X_SAVE_UNIT = "s"
    X_SAVE_SCALE = 1e-6

    def __init__(self, config):
        super().__init__(config)
        self.iq_list = None
        self.t = None

    def _normalized_cfg(self):
        cfg = dict(self.cfg)

        def _first_value(value, default=None):
            if value is None:
                return default
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                values = [v for v in value if v is not None]
                return values[0] if values else default
            return value

        if "ro_len" not in cfg and "ro_length" in cfg:
            cfg["ro_len"] = cfg["ro_length"]
        if "res_len" not in cfg and "res_length" in cfg:
            cfg["res_len"] = cfg["res_length"]
        cfg["ro_len"] = _first_value(cfg.get("ro_len"), 1)
        cfg["res_len"] = _first_value(cfg.get("res_len"), cfg["ro_len"])
        cfg["relax_delay"] = _first_value(cfg.get("relax_delay"), 0)
        cfg["nqz_res"] = _first_value(cfg.get("mux_nqz", cfg.get("nqz_res")), 1)
        if "active_ro_chs" not in cfg:
            cfg["active_ro_chs"] = cfg.get("ro_chs", [])
        if "trigger_ro_chs" not in cfg:
            cfg["trigger_ro_chs"] = cfg["active_ro_chs"]
        return cfg

    def _create_program(self):
        cfg = self._normalized_cfg()
        return MuxTOFProgram(
            self.soccfg,
            reps=1,
            final_delay=cfg.get("relax_delay", 0),
            cfg=cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_time_axis(ro_index=0)

    def run(self, py_avg=1, **kwargs) -> ExperimentData:
        return self.liveplot(py_avg=py_avg, **kwargs)

    def liveplot(self, py_avg=1, threshold=1.5, plot=True) -> ExperimentData:
        cfg = self._normalized_cfg()
        prog = self._create_program()
        self._last_prog = prog
        self.t = self._extract_sweep_axis(prog)
        self._sweep_vals_x = self.t

        active_ro_chs = list(cfg.get("active_ro_chs", cfg["ro_chs"]))
        qubit_names = list(cfg.get("qubit_names", [f"ro{ch}" for ch in active_ro_chs]))
        trace_count = len(active_ro_chs)
        iq_sum = None
        interrupted = False

        if plot:
            fig, axes = plt.subplots(trace_count, 1, figsize=(8, max(3, 2.5 * trace_count)), squeeze=False)
            axes = axes[:, 0]
            lines = []
            for ax, name in zip(axes, qubit_names):
                (line,) = ax.plot(self.t, np.full_like(self.t, np.nan, dtype=float), alpha=0.85)
                ax.set_ylabel(name)
                ax.set_xlim(np.min(self.t), np.max(self.t))
                lines.append(line)
            axes[-1].set_xlabel(self.X_LABEL)
            title = axes[0].set_title(f"{self.TITLE_PREFIX} | Average: 0 / 0")
            plot_id = f"live-plot-mux-tof-{np.random.randint(int(1e9))}"
            display(fig, display_id=plot_id)
        else:
            fig = axes = lines = title = plot_id = None

        avg_done = 0
        try:
            for i in tqdm(range(py_avg), desc="Software Average Count"):
                self.iq_list = prog.acquire_decimated(self.soc, rounds=1, progress=False)
                current = np.asarray([trace.dot([1, 1j]) for trace in self.iq_list[:trace_count]])
                iq_sum = current if iq_sum is None else iq_sum + current
                self.iqdata = iq_sum / (i + 1)
                avg_done = i + 1

                if plot:
                    for ax, line, trace in zip(axes, lines, self.iqdata):
                        plot_data = np.abs(trace)
                        line.set_ydata(plot_data)
                        cmin, cmax = np.min(plot_data), np.max(plot_data)
                        span = max(cmax - cmin, 1e-9)
                        ax.set_ylim(cmin - 0.1 * span, cmax + 0.1 * span)
                    title.set_text(f"{self.TITLE_PREFIX} | Average: {avg_done} / {py_avg}")
                    update_display(fig, display_id=plot_id)
        except KeyboardInterrupt:
            interrupted = True

        if plot:
            clear_output(wait=True)
            if fig is not None:
                plt.close(fig)

        figures = []
        trig_times = {}
        if self.iqdata is not None and plot:
            final_fig, final_axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 2.5 * trace_count)), squeeze=False
            )
            final_axes = final_axes[:, 0]
            for ax, name, trace in zip(final_axes, qubit_names, self.iqdata):
                mag = np.abs(trace)
                ax.plot(self.t, mag, "o-", markersize=2)
                mean = np.mean(mag)
                cross_idx = np.argmax(mag > threshold * mean)
                trig_time = float(self.t[cross_idx])
                trig_times[name] = trig_time
                ax.axvline(trig_time, c="r", ls="--", label=f"TOF: {trig_time:.2f} us")
                ax.set_ylabel(name)
                ax.set_xlim(np.min(self.t), np.max(self.t))
                ax.legend()
            final_axes[-1].set_xlabel(self.X_LABEL)
            final_axes[0].set_title(self.TITLE_PREFIX + (" (Interrupted)" if interrupted else ""))
            display(final_fig)
            figures.append(final_fig)
            plt.close(final_fig)

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.t,
            fit_result={name: (tof, None) for name, tof in trig_times.items()},
            config=clean_config(cfg),
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "mask": list(cfg["mask"]),
                "LO_ext": cfg.get("LO_ext"),
            },
            figures=figures,
            quality=QualityFlag.GOOD if self.iqdata is not None else QualityFlag.BAD,
            quality_message="Mux TOF acquired." if self.iqdata is not None else "No data acquired.",
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            interrupted=interrupted,
            avg_count=avg_done,
        )
        self.result = result
        return result
