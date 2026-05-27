"""
Mux single-shot ge readout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag
from ..tools.system_tool import clean_config


class MuxSingleShotGEProgram(AveragerProgramV2):
    """Mux single-shot ge: read g, prepare e with pi pulses, read e."""

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

        self.add_loop("shotloop", cfg["shots"])

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
            self._add_pi_pulse(cfg, idx, cfg["qb_ch"][idx], f"{name}_pi")

    def _add_pi_pulse(self, cfg, idx, qb_ch, pulse_name):
        pulse_type = cfg["pulse_type"][idx]
        gain = cfg.get("qb_pi_gain", cfg.get("pi_gain_ge"))[idx]
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
                phase=cfg["qb_phase"][idx],
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
                phase=cfg["qb_phase"][idx],
                gain=gain,
            )

    def _body(self, cfg):
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)

        self.delay_auto(cfg["relax_delay"], tag="relax_wait")
        for idx, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][idx], name=f"{name}_pi", t=0)
        self.delay_auto(0.01, tag="wait")

        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxSingleShotGE(BaseExperiment):
    """Mux single-shot ge readout and simple threshold analysis."""

    EXPT_NAME = "s000_mux_singleshot_ge"
    TAG = "MuxSingleShotGE"
    X_LABEL = "I [ADC units]"
    Y_LABEL = "Q [ADC units]"
    TITLE_PREFIX = "Mux SingleShot ge"

    def __init__(self, config):
        super().__init__(config)
        self.data = None

    def _create_program(self):
        cfg = dict(self.cfg)
        return MuxSingleShotGEProgram(
            self.soccfg,
            reps=1,
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )

    def _extract_sweep_axis(self, prog):
        return np.arange(int(self.cfg["shots"]))

    @staticmethod
    def _extract_iq(iq_list, n_trace):
        data = np.full((n_trace, 2, iq_list[0].shape[1]), np.nan + 1j * np.nan, dtype=complex)
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx])
            data[idx, 0, :] = arr[0, :, 0] + 1j * arr[0, :, 1]
            data[idx, 1, :] = arr[1, :, 0] + 1j * arr[1, :, 1]
        return data

    @staticmethod
    def _analyze_ge(g_iq, e_iq):
        g_mean = np.mean(g_iq)
        e_mean = np.mean(e_iq)
        theta = -np.arctan2((e_mean - g_mean).imag, (e_mean - g_mean).real)

        def rotate(values):
            return values.real * np.cos(theta) - values.imag * np.sin(theta)

        g_i = rotate(g_iq)
        e_i = rotate(e_iq)
        g_center = float(np.mean(g_i))
        e_center = float(np.mean(e_i))
        threshold = 0.5 * (g_center + e_center)

        if e_center >= g_center:
            p_g = float(np.mean(g_i < threshold))
            p_e = float(np.mean(e_i > threshold))
        else:
            p_g = float(np.mean(g_i > threshold))
            p_e = float(np.mean(e_i < threshold))

        fidelity = 0.5 * (p_g + p_e)
        snr = abs(e_center - g_center) / np.sqrt(np.var(g_i) + np.var(e_i))
        return {
            "theta_deg": float(theta * 180 / np.pi),
            "threshold": float(threshold),
            "fidelity": float(fidelity),
            "snr": float(snr),
            "g_center": g_center,
            "e_center": e_center,
            "p_g": p_g,
            "p_e": p_e,
            "g_i": g_i,
            "e_i": e_i,
        }

    def run(self, shots=None, plot=True):
        cfg = dict(self.cfg)
        if shots is not None:
            cfg["shots"] = int(shots)

        prog = MuxSingleShotGEProgram(
            self.soccfg,
            reps=1,
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )
        self._last_prog = prog

        qubit_names = list(cfg["qubit_names"])
        active_ro_chs = list(cfg["active_ro_chs"])
        active_slots = list(cfg["active_slots"])
        trace_count = len(active_ro_chs)

        interrupted = False
        try:
            iq_list = prog.acquire(self.soc, rounds=1, progress=True)
            self.iqdata = self._extract_iq(iq_list, trace_count)
        except KeyboardInterrupt:
            interrupted = True
            self.iqdata = None

        fit_result = {}
        analysis = {}
        if self.iqdata is not None and np.isfinite(self.iqdata).any():
            for idx, name in enumerate(qubit_names):
                stats = self._analyze_ge(self.iqdata[idx, 0], self.iqdata[idx, 1])
                analysis[name] = {
                    key: val
                    for key, val in stats.items()
                    if key not in {"g_i", "e_i"}
                }
                fit_result[f"{name}_fidelity"] = (round(stats["fidelity"], 6), None)
                fit_result[f"{name}_threshold"] = (round(stats["threshold"], 6), None)
                fit_result[f"{name}_rotation_deg"] = (round(stats["theta_deg"], 6), None)
                fit_result[f"{name}_snr"] = (round(stats["snr"], 6), None)

        self.data = {}
        if self.iqdata is not None:
            for idx, name in enumerate(qubit_names):
                self.data[name] = {
                    "Ig": self.iqdata[idx, 0].real,
                    "Qg": self.iqdata[idx, 0].imag,
                    "Ie": self.iqdata[idx, 1].real,
                    "Qe": self.iqdata[idx, 1].imag,
                }

        figures = []
        if plot and self.iqdata is not None and np.isfinite(self.iqdata).any():
            for idx, name in enumerate(qubit_names):
                stats = self._analyze_ge(self.iqdata[idx, 0], self.iqdata[idx, 1])
                fig, axes = plt.subplots(1, 2, figsize=(9, 4))
                axes[0].scatter(self.iqdata[idx, 0].real, self.iqdata[idx, 0].imag, s=6, alpha=0.45, label="g")
                axes[0].scatter(self.iqdata[idx, 1].real, self.iqdata[idx, 1].imag, s=6, alpha=0.45, label="e")
                axes[0].set_xlabel("I")
                axes[0].set_ylabel("Q")
                axes[0].axis("equal")
                axes[0].legend()

                bins = int(cfg.get("bins", 120))
                axes[1].hist(stats["g_i"], bins=bins, alpha=0.55, density=True, label="g")
                axes[1].hist(stats["e_i"], bins=bins, alpha=0.55, density=True, label="e")
                axes[1].axvline(stats["threshold"], color="k", linestyle="--", label="threshold")
                axes[1].set_xlabel("Rotated I")
                axes[1].set_ylabel("Density")
                axes[1].legend()
                fig.suptitle(f"{name} | F = {stats['fidelity']:.4f}, SNR = {stats['snr']:.3f}")
                fig.tight_layout()
                figures.append(fig)
                plt.show()

        has_data = self.iqdata is not None and np.isfinite(self.iqdata).any()
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=np.arange(int(cfg["shots"])),
            fit_result=fit_result,
            config=clean_config(cfg),
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "analysis": analysis,
                "states": ["g", "e"],
                "shots": int(cfg["shots"]),
            },
            figures=figures,
            quality=QualityFlag.GOOD if has_data else QualityFlag.BAD,
            quality_message="Mux single-shot ge acquired." if has_data else "No data acquired.",
            interrupted=interrupted,
            avg_count=1,
        )
        self.result = result
        return result


__all__ = ["MuxSingleShotGE", "MuxSingleShotGEProgram"]
