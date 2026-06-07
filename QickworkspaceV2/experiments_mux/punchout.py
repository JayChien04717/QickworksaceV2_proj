"""
Mux resonator punchout using software gain and frequency loops.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag
from ..tools.system_tool import clean_config


class MuxPunchoutProgram(AveragerProgramV2):
    """Single mux readout point for resonator punchout."""

    def _initialize(self, cfg):
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
        for ch, slot in zip(ro_chs, cfg["active_slots"]):
            self.declare_readout(
                ch=ch,
                length=cfg["ro_length"],
                freq=cfg["res_freqs"][slot],
                phase=cfg["ro_phases"][slot],
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
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxPunchout(BaseExperiment):
    """Mux resonator punchout with one 2D map per armed qubit."""

    EXPT_NAME = "s002b_mux_res_punchout_ge"
    TAG = "MuxPunchout"
    X_LABEL = "Frequency (MHz)"
    Y_LABEL = "Readout gain"
    TITLE_PREFIX = "Mux Resonator Punchout"
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6
    Y_SAVE_NAME = "DAC Gains"
    Y_SAVE_UNIT = "a.u."
    Y_SAVE_SCALE = 1.0

    def __init__(self, config):
        super().__init__(config)
        self.freq_offsets = None
        self.gain_axis = None
        self.freq_axes = None

    @staticmethod
    def _point_iq(iq_list, n_trace):
        vals = []
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx][0]).squeeze()
            if arr.ndim > 0 and arr.shape[-1] == 2:
                vals.append(np.asarray(arr).dot([1, 1j]).reshape(-1)[0])
            else:
                vals.append(np.asarray(arr, dtype=complex).reshape(-1)[0])
        return np.asarray(vals, dtype=complex)

    @staticmethod
    def _configured_axis(cfg, key, fallback):
        if key in cfg:
            arr = np.asarray(cfg[key], dtype=float)
            if arr.ndim == 1 and arr.size > 1:
                return arr
        return fallback

    @staticmethod
    def _point_cfg(cfg, active_slots, freq_offset, gain):
        point_cfg = dict(cfg)
        point_cfg["res_freqs"] = [float(freq) + freq_offset for freq in cfg["res_freqs"]]
        point_cfg["res_gains"] = list(cfg["res_gains"])
        for slot in active_slots:
            point_cfg["res_gains"][slot] = float(gain)
        return point_cfg

    def run(
        self,
        py_avg=1,
        span=20.0,
        f_steps=101,
        gains=None,
        iq_process="abs",
        normalize_per_power=True,
        plot=False,
    ):
        cfg = dict(self.cfg)
        active_slots = list(cfg["active_slots"])
        active_ro_chs = list(cfg["active_ro_chs"])
        qubit_names = list(cfg["qubit_names"])
        trace_count = len(active_ro_chs)

        half_span = abs(float(span)) / 2.0
        self.freq_offsets = self._configured_axis(
            cfg, "freq_offsets", np.linspace(-half_span, half_span, int(f_steps))
        )
        if gains is None:
            gains = cfg["gain_axis"] if "gain_axis" in cfg else np.linspace(0.0, 1.0, 21)
        self.gain_axis = np.asarray(gains, dtype=float)

        lo_ext = cfg.get("LO_ext") or 0
        centers_if = np.asarray([cfg["res_freqs"][slot] for slot in active_slots])
        centers_abs = centers_if + float(lo_ext)
        self.freq_axes = {
            name: (center + self.freq_offsets).tolist()
            for name, center in zip(qubit_names, centers_abs)
        }

        self.iqdata = np.full(
            (trace_count, len(self.gain_axis), len(self.freq_offsets)),
            np.nan + 1j * np.nan,
            dtype=complex,
        )
        interrupted = False
        points_done = 0

        try:
            for g_idx, gain in enumerate(tqdm(self.gain_axis, desc="Gain")):
                for f_idx, offset in enumerate(tqdm(self.freq_offsets, desc="Frequency", leave=False)):
                    point_cfg = self._point_cfg(cfg, active_slots, float(offset), float(gain))
                    prog = MuxPunchoutProgram(
                        self.soccfg,
                        reps=point_cfg["reps"],
                        final_delay=point_cfg["relax_delay"],
                        cfg=point_cfg,
                    )
                    self._last_prog = prog
                    iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
                    self.iqdata[:, g_idx, f_idx] = self._point_iq(iq_list, trace_count)
                    points_done += 1
        except KeyboardInterrupt:
            interrupted = True

        figures = []
        plot_data = self._process_plot_data(self.iqdata, iq_process)
        display_data = (
            self._normalize_per_power(plot_data) if normalize_per_power else plot_data
        )
        if plot and self.iqdata is not None and np.isfinite(self.iqdata).any():
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 3 * trace_count)), squeeze=False
            )
            for ax, name, center, trace in zip(axes[:, 0], qubit_names, centers_abs, display_data):
                x = center + self.freq_offsets
                mesh = ax.pcolormesh(x, self.gain_axis, trace, shading="auto")
                ax.set_ylabel(f"{name} gain")
                ax.set_xlabel(self.X_LABEL)
                cbar_label = f"{iq_process} normalized per power" if normalize_per_power else iq_process
                fig.colorbar(mesh, ax=ax, label=cbar_label)
            axes[0, 0].set_title(
                self.TITLE_PREFIX + (" (Interrupted)" if interrupted else "")
            )
            fig.tight_layout()
            figures.append(fig)
            plt.show()

        has_data = self.iqdata is not None and np.isfinite(self.iqdata).any()
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.freq_offsets,
            y_axis=self.gain_axis,
            fit_result={"status": ("punchout_acquired", None)} if has_data else {},
            config=clean_config(cfg),
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "center_freqs_mhz": centers_abs.tolist(),
                "frequency_axes_mhz": self.freq_axes,
                "gain_axis": self.gain_axis.tolist(),
                "normalize_per_power": bool(normalize_per_power),
                "points_acquired": points_done,
                "LO_ext": cfg.get("LO_ext"),
            },
            figures=figures,
            quality=QualityFlag.GOOD if has_data else QualityFlag.BAD,
            quality_message="Mux punchout acquired." if has_data else "No data acquired.",
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result

    @staticmethod
    def _process_plot_data(iqdata, iq_process):
        iq_process = (iq_process or "abs").lower()
        if iq_process in {"real", "i", "avgi"}:
            return np.real(iqdata)
        if iq_process in {"imag", "q", "avgq"}:
            return np.imag(iqdata)
        if iq_process == "phase":
            return np.unwrap(np.angle(iqdata), axis=-1)
        return np.abs(iqdata)

    @staticmethod
    def _normalize_per_power(plot_data):
        data = np.asarray(plot_data, dtype=float)
        row_min = np.nanmin(data, axis=-1, keepdims=True)
        row_max = np.nanmax(data, axis=-1, keepdims=True)
        row_span = row_max - row_min
        normalized = (data - row_min) / np.where(row_span > 0, row_span, 1.0)
        return normalized


__all__ = ["MuxPunchout", "MuxPunchoutProgram"]
