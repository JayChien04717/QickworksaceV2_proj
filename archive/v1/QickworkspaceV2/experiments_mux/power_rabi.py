"""
Mux power Rabi using a QICK gain loop.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2, QickSweep1D

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag
from ..tools.system_tool import clean_config


class MuxPowerRabiProgram(AveragerProgramV2):
    """Mux power Rabi: one QICK gainloop, mux readout on active channels."""

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
        for ch, idx in zip(ro_chs, cfg["active_slots"]):
            self.declare_readout(
                ch=ch,
                length=cfg["ro_length"],
                freq=cfg["res_freqs"][idx],
                phase=cfg["ro_phases"][idx],
                gen_ch=res_ch,
            )

        self.add_loop("gainloop", cfg["steps"])

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

        gain = QickSweep1D("gainloop", cfg["start_gain"], cfg["stop_gain"])
        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            qb_ch = cfg["qb_ch"][idx]
            pulse_name = f"{name}_pulse"
            pulse_type = cfg["pulse_type"][idx]

            if pulse_type == "const":
                self.add_pulse(
                    ch=qb_ch,
                    name=pulse_name,
                    style="const",
                    length=cfg["qb_flat_top_length_ge"][idx],
                    freq=cfg["qb_freq_ge"][idx],
                    phase=cfg["qb_phase"][idx],
                    gain=gain,
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
                    freq=cfg["qb_freq_ge"][idx],
                    phase=cfg["qb_phase"][idx],
                    gain=gain,
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
                    freq=cfg["qb_freq_ge"][idx],
                    phase=cfg["qb_phase"][idx],
                    gain=gain,
                )

    def _body(self, cfg):
        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][idx], name=f"{name}_pulse", t=0)
        self.delay_auto(0.05)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxPowerRabi(BaseExperiment):
    """Mux power Rabi with one gain sweep for all armed qubits."""

    EXPT_NAME = "s005_mux_power_rabi_ge"
    TAG = "MuxPowerRabi"
    X_LABEL = "Gain (a.u.)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Mux Power Rabi"
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    def __init__(self, config):
        super().__init__(config)
        self.gain_axis = None

    def _create_program(self):
        cfg = dict(self.cfg)
        return MuxPowerRabiProgram(
            self.soccfg,
            reps=cfg["reps"],
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )

    def _extract_sweep_axis(self, prog):
        return self.gain_axis

    @staticmethod
    def _sweep_iq(iq_list, n_trace):
        vals = []
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx][0]).squeeze()
            if arr.ndim > 0 and arr.shape[-1] == 2:
                vals.append(np.asarray(arr).dot([1, 1j]).reshape(-1))
            else:
                vals.append(np.asarray(arr, dtype=complex).reshape(-1))
        return np.asarray(vals, dtype=complex)

    def run(self, py_avg=1, iq_process="abs", plot=False):
        cfg = dict(self.cfg)
        prog = MuxPowerRabiProgram(
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
        self.gain_axis = BaseExperiment._resolve_axis(
            prog.get_pulse_param(f"{qubit_names[0]}_pulse", "gain", as_array=True),
            cfg["steps"],
        )

        interrupted = False
        try:
            iq_list = prog.acquire(self.soc, rounds=py_avg, progress=True)
            self.iqdata = self._sweep_iq(iq_list, trace_count)
        except KeyboardInterrupt:
            interrupted = True
            self.iqdata = None

        fit_result = {}
        fit_method = {}
        fit_params = {}
        if self.iqdata is not None and np.isfinite(self.iqdata).any():
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            for idx, name in enumerate(qubit_names):
                trace = plot_data[idx]
                finite = np.isfinite(trace)
                if not np.any(finite):
                    continue
                fit = self._fit_power_rabi(self.gain_axis[finite], trace[finite])
                if fit is None:
                    continue
                pi_gain, pi2_gain, popt = fit
                fit_result[f"{name}_pi_gain"] = (round(float(pi_gain), 6), None)
                fit_result[f"{name}_pi2_gain"] = (round(float(pi2_gain), 6), None)
                fit_method[name] = "sinusoid"
                fit_params[name] = np.asarray(popt, dtype=float).tolist()

        figures = []
        if plot and self.iqdata is not None and np.isfinite(self.iqdata).any():
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 2.5 * trace_count)), squeeze=False
            )
            for ax, name, trace in zip(axes[:, 0], qubit_names, plot_data):
                ax.plot(self.gain_axis, trace, "o-", markersize=3)
                pi_gain = fit_result.get(f"{name}_pi_gain", (None,))[0]
                pi2_gain = fit_result.get(f"{name}_pi2_gain", (None,))[0]
                if pi_gain is not None:
                    ax.axvline(pi_gain, c="r", ls="--", label=f"pi {pi_gain:.4f}")
                if pi2_gain is not None:
                    ax.axvline(pi2_gain, c="g", ls="--", label=f"pi/2 {pi2_gain:.4f}")
                if pi_gain is not None or pi2_gain is not None:
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
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.gain_axis,
            fit_result=fit_result,
            config=clean_config(cfg),
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "fit_method": fit_method,
                "fit_params": fit_params,
                "points_acquired": int(cfg["steps"]) if has_data else 0,
            },
            figures=figures,
            quality=QualityFlag.GOOD if has_data else QualityFlag.BAD,
            quality_message="Mux power Rabi acquired."
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
    def _fit_power_rabi(gain_axis, trace):
        try:
            from ..tools.fitting import fitsin, fix_phase

            popt, pcov, _ = fitsin(gain_axis, trace)
            pi_gain, pi2_gain = fix_phase(popt)
            if not np.isfinite(pi_gain) or not np.isfinite(pi2_gain):
                raise RuntimeError("sinusoid fit returned invalid gains")
            return pi_gain, pi2_gain, popt
        except Exception:
            return None

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


__all__ = ["MuxPowerRabi", "MuxPowerRabiProgram"]
