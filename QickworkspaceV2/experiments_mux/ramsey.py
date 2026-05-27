"""
Mux Ramsey T2* using a QICK wait loop.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2, QickSweep1D

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag
from ..tools.system_tool import clean_config


class MuxRamseyProgram(AveragerProgramV2):
    """Mux Ramsey: two pi/2 pulses, swept wait, mux readout."""

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

        self.add_loop("waitloop", cfg["steps"])
        self.wait_time = QickSweep1D("waitloop", cfg["start_wait"], cfg["stop_wait"])

        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.declare_gen(ch=cfg["qb_ch"][idx], nqz=cfg["nqz_qb"][idx])

        self.add_pulse(
            ch=res_ch,
            name="mux_readout",
            style="const",
            length=cfg["res_length"],
            mask=cfg["mask"],
        )

        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            qb_ch = cfg["qb_ch"][idx]
            pulse_type = cfg["pulse_type"][idx]
            ramsey_freq = cfg["ramsey_freq"][idx]
            phase1 = cfg["qb_phase"][idx]
            phase2 = phase1 + self.wait_time * 360 * ramsey_freq

            self._add_qubit_pulse(cfg, idx, qb_ch, f"{name}_pulse1", pulse_type, phase1)
            self._add_qubit_pulse(cfg, idx, qb_ch, f"{name}_pulse2", pulse_type, phase2)

    def _add_qubit_pulse(self, cfg, idx, qb_ch, pulse_name, pulse_type, phase):
        gain = cfg["pi2_gain_ge"][idx]
        if pulse_type == "const":
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="const",
                length=cfg["qb_flat_top_length_ge"][idx],
                freq=cfg["qb_freq_ge"][idx],
                phase=phase,
                gain=gain,
            )
        elif pulse_type == "arb":
            env_name = f"{pulse_name}_envelope"
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
                phase=phase,
                gain=gain,
            )
        else:
            env_name = f"{pulse_name}_envelope"
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
                phase=phase,
                gain=gain,
            )

    def _body(self, cfg):
        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][idx], name=f"{name}_pulse1", t=0)
        self.delay_auto(self.wait_time + 0.01, tag="wait")
        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][idx], name=f"{name}_pulse2", t=0)
        self.delay_auto(0.05)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxRamsey(BaseExperiment):
    """Mux Ramsey T2* with one wait sweep for all armed qubits."""

    EXPT_NAME = "s006_mux_ramsey_ge"
    TAG = "MuxRamsey"
    X_LABEL = "Delay time (us)"
    Y_LABEL = "ADC Units"
    TITLE_PREFIX = "Mux Ramsey"
    X_SAVE_NAME = "Delay time"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def __init__(self, config):
        super().__init__(config)
        self.wait_axis = None

    def _create_program(self):
        cfg = dict(self.cfg)
        return MuxRamseyProgram(
            self.soccfg,
            reps=cfg["reps"],
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )

    def _extract_sweep_axis(self, prog):
        return self.wait_axis

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
        prog = MuxRamseyProgram(
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
        self.wait_axis = MuxRamsey._get_wait_axis(prog, cfg)

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
                slot = active_slots[idx]
                trace = plot_data[idx]
                finite = np.isfinite(trace)
                if not np.any(finite):
                    continue
                fit = self._fit_ramsey(
                    self.wait_axis[finite],
                    trace[finite],
                    cfg["ramsey_freq"][slot],
                    cfg["qb_freq_ge"][slot],
                )
                if fit is None:
                    continue
                t2r, fit_freq, detune, corrected, popt, perr, method = fit
                fit_result[f"{name}_T2r_us"] = (
                    round(float(t2r), 6),
                    perr[3] if perr is not None and len(perr) > 3 else None,
                )
                fit_result[f"{name}_fit_freq_MHz"] = (
                    round(float(fit_freq), 6),
                    perr[1] if perr is not None and len(perr) > 1 else None,
                )
                fit_result[f"{name}_detune_MHz"] = (
                    round(float(detune), 6),
                    None,
                )
                fit_result[f"{name}_corrected_freq_MHz"] = (
                    round(float(corrected), 6),
                    None,
                )
                fit_method[name] = method
                fit_params[name] = np.asarray(popt, dtype=float).tolist()

        figures = []
        if plot and self.iqdata is not None and np.isfinite(self.iqdata).any():
            plot_data = self._process_plot_data(self.iqdata, iq_process)
            fig, axes = plt.subplots(
                trace_count, 1, figsize=(8, max(3, 2.5 * trace_count)), squeeze=False
            )
            from ..tools.fitting import decaysin, expfunc

            for ax, name, trace in zip(axes[:, 0], qubit_names, plot_data):
                ax.plot(self.wait_axis, trace, "o-", markersize=3)
                method = fit_method.get(name)
                if name in fit_params:
                    fit_func = decaysin if method == "decaysin" else expfunc
                    ax.plot(
                        self.wait_axis,
                        fit_func(self.wait_axis, *fit_params[name]),
                        "-",
                        linewidth=2,
                        label="fit",
                    )
                t2r = fit_result.get(f"{name}_T2r_us", (None,))[0]
                if t2r is not None:
                    ax.set_title(f"{name} | T2* = {t2r:.3f} us")
                else:
                    ax.set_title(name)
                ax.set_ylabel(name)
                ax.set_xlabel(self.X_LABEL)
                if name in fit_params:
                    ax.legend()
            fig.tight_layout()
            figures.append(fig)
            plt.show()

        has_data = self.iqdata is not None and np.isfinite(self.iqdata).any()
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.wait_axis,
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
            quality_message="Mux Ramsey acquired." if has_data else "No data acquired.",
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result

    def correct_detune(self):
        """Correct each armed qubit frequency from mux Ramsey detuning fits."""
        if self.result is None:
            raise RuntimeError("Run the experiment first.")

        cfg = self.cfg
        qubit_names = list(cfg["qubit_names"])
        active_slots = list(cfg["active_slots"])
        qb_freqs = list(cfg["qb_freq_ge"])
        corrections = {}

        for name, slot in zip(qubit_names, active_slots):
            detune = self.result.fit_result.get(f"{name}_detune_MHz", (None,))[0]
            if detune is None:
                print(f"{name}: detune not available.")
                continue

            old_freq = float(qb_freqs[slot])
            delta = round(float(detune), 2)
            if abs(float(detune)) > 0.005:
                new_freq = old_freq - delta
                qb_freqs[slot] = new_freq
                status = "corrected"
                print(f"{name}: detune error {delta:+.5f} MHz, qb_freq_ge -> {new_freq:.5f} MHz")
            else:
                new_freq = old_freq
                status = "detune < 5 kHz"
                print(f"{name}: detune < 5 kHz")

            corrections[name] = {
                "old_qb_freq_ge": old_freq,
                "new_qb_freq_ge": new_freq,
                "detune_MHz": float(detune),
                "ramsey_freq_MHz": float(cfg["ramsey_freq"][slot]),
                "detune_error_MHz": delta,
                "status": status,
            }

        cfg["qb_freq_ge"] = qb_freqs
        return corrections

    @staticmethod
    def _fit_ramsey(wait_axis, trace, ramsey_freq, qb_freq):
        try:
            if ramsey_freq:
                from ..tools.fitting import fitdecaysin

                popt, pcov, _ = fitdecaysin(wait_axis, trace)
                perr = np.sqrt(np.abs(np.diag(pcov)))
                t2r = abs(float(popt[3]))
                fit_freq = float(popt[1])
                detune = fit_freq - float(ramsey_freq)
                corrected = float(qb_freq) - round(detune, 2)
                return t2r, fit_freq, detune, corrected, popt, perr, "decaysin"

            from ..tools.fitting import fitexp

            popt, pcov, _ = fitexp(wait_axis, trace)
            perr = np.sqrt(np.abs(np.diag(pcov)))
            t2r = abs(float(popt[2]))
            return t2r, 0.0, 0.0, float(qb_freq), popt, perr, "exponential"
        except Exception:
            return None

    @staticmethod
    def _get_wait_axis(prog, cfg):
        try:
            wait_axis = np.asarray(
                prog.get_time_param("wait", "t", as_array=True),
                dtype=float,
            ).reshape(-1)
            if wait_axis.size == int(cfg["steps"]):
                return wait_axis
        except Exception:
            pass
        return np.linspace(
            float(cfg["start_wait"]),
            float(cfg["stop_wait"]),
            int(cfg["steps"]),
        )

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


__all__ = ["MuxRamsey", "MuxRamseyProgram"]
