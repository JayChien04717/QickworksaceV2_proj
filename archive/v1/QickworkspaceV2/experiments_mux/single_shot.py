"""
Mux single-shot ge readout.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.experiment_data import ExperimentData, QualityFlag


class MuxSingleShotGEProgram(AveragerProgramV2):
    """Mux single-shot ge: read g, prepare e with pi pulses, read e."""

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
        """Add pi pulse.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        idx : Any
            Value for ``idx``.
        qb_ch : Any
            Value for ``qb_ch``.
        pulse_name : Any
            Name of the pulse.
        """
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
        """Execute one iteration of the pulse sequence.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
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
        """Initialize the MuxSingleShotGE instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.data = None

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        return MuxSingleShotGEProgram(
            self.soccfg,
            reps=1,
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
        return np.arange(int(self.cfg["shots"]))

    @staticmethod
    def _extract_iq(iq_list, n_trace):
        """Extract iq.

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
        data = np.full((n_trace, 2, iq_list[0].shape[1]), np.nan + 1j * np.nan, dtype=complex)
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx])
            data[idx, 0, :] = arr[0, :, 0] + 1j * arr[0, :, 1]
            data[idx, 1, :] = arr[1, :, 0] + 1j * arr[1, :, 1]
        return data

    @staticmethod
    def _analyze_ge(g_iq, e_iq):
        """Return the analyze ge result.

        Parameters
        ----------
        g_iq : Any
            Value for ``g_iq``.
        e_iq : Any
            Value for ``e_iq``.

        Returns
        -------
        Any
            Result of the operation.
        """
        g_mean = np.mean(g_iq)
        e_mean = np.mean(e_iq)
        theta = -np.arctan2((e_mean - g_mean).imag, (e_mean - g_mean).real)

        def rotate(values):
            """Return the rotate result.

            Parameters
            ----------
            values : Any
                Values to process.

            Returns
            -------
            Any
                Result of the operation.
            """
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
        """Run the operation.

        Parameters
        ----------
        shots : Any, default: None
            Value for ``shots``.
        plot : Any, default: True
            Value for ``plot``.

        Returns
        -------
        Any
            Result of the operation.
        """
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
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "analysis": analysis,
                "states": ["g", "e"],
                "shots": int(cfg["shots"]),
            },
            axes={
                "qubit": {"values": qubit_names},
                "state": {"values": ["g", "e"]},
                "shot": {"values": np.arange(int(cfg["shots"])), "unit": "#"},
            },
            dataset_dims={"iq": ["qubit", "state", "shot"]},
            analysis_data={
                "fidelity": {
                    "values": np.asarray([analysis.get(name, {}).get("fidelity", np.nan) for name in qubit_names]),
                    "dims": ["qubit"],
                },
                "threshold": {
                    "values": np.asarray([analysis.get(name, {}).get("threshold", np.nan) for name in qubit_names]),
                    "dims": ["qubit"],
                },
                "rotation_deg": {
                    "values": np.asarray([analysis.get(name, {}).get("theta_deg", np.nan) for name in qubit_names]),
                    "dims": ["qubit"],
                },
                "snr": {
                    "values": np.asarray([analysis.get(name, {}).get("snr", np.nan) for name in qubit_names]),
                    "dims": ["qubit"],
                },
            },
            data_kind="single_shot",
            analysis_id="single_shot",
            plot_id="single_shot_iq",
            figures=figures,
            quality=QualityFlag.GOOD if has_data else QualityFlag.BAD,
            quality_message="Mux single-shot ge acquired." if has_data else "No data acquired.",
            interrupted=interrupted,
            avg_count=1,
        )
        self.result = result
        return result


class MuxSingleShotGEOpt(BaseExperiment):
    """Grid optimize mux single-shot ge readout length, gain, and frequency offset."""

    EXPT_NAME = "s000_mux_singleshot_ge_opt"
    TAG = "MuxSingleShotGEOpt"
    TITLE_PREFIX = "Mux SingleShot ge Optimize"

    def __init__(self, config):
        """Initialize the MuxSingleShotGEOpt instance.

        Parameters
        ----------
        config : Any
            Experiment configuration.
        """
        super().__init__(config)
        self.length_axis = None
        self.gain_axis = None
        self.freq_offsets = None
        self.metric_arrays = None
        self.best = None

    @staticmethod
    def _axis(value, default):
        """Return the axis result.

        Parameters
        ----------
        value : Any
            Value to apply.
        default : Any
            Value for ``default``.

        Returns
        -------
        Any
            Result of the operation.
        """
        if value is None:
            value = default
        if isinstance(value, (list, tuple, np.ndarray)):
            return np.asarray(value, dtype=float)
        return np.asarray([value], dtype=float)

    @staticmethod
    def _point_cfg(cfg, active_slots, length, gain, freq_offset, shots):
        """Return the point cfg result.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        active_slots : Any
            Value for ``active_slots``.
        length : Any
            Value for ``length``.
        gain : Any
            Value for ``gain``.
        freq_offset : Any
            Value for ``freq_offset``.
        shots : Any
            Value for ``shots``.

        Returns
        -------
        Any
            Result of the operation.
        """
        point_cfg = dict(cfg)
        point_cfg["shots"] = int(shots)

        if length is not None:
            point_cfg["res_length"] = float(length)
            point_cfg["ro_length"] = float(length)

        if gain is not None:
            point_cfg["res_gains"] = list(cfg["res_gains"])
            for slot in active_slots:
                point_cfg["res_gains"][slot] = float(gain)

        if freq_offset is not None:
            point_cfg["res_freqs"] = list(cfg["res_freqs"])
            for slot in active_slots:
                point_cfg["res_freqs"][slot] = float(cfg["res_freqs"][slot]) + float(freq_offset)
            active_freqs = [point_cfg["res_freqs"][slot] for slot in active_slots]
            if active_freqs:
                point_cfg["mixer_freq"] = int(round(float(np.mean(active_freqs))))

        return point_cfg

    @staticmethod
    def _result_arrays(n_qubits, n_l, n_g, n_f):
        """Return the result arrays result.

        Parameters
        ----------
        n_qubits : Any
            Value for ``n_qubits``.
        n_l : Any
            Value for ``n_l``.
        n_g : Any
            Value for ``n_g``.
        n_f : Any
            Value for ``n_f``.

        Returns
        -------
        Any
            Result of the operation.
        """
        shape = (n_qubits, n_l, n_g, n_f)
        return {
            "fidelity": np.full(shape, np.nan, dtype=float),
            "threshold": np.full(shape, np.nan, dtype=float),
            "rotation_deg": np.full(shape, np.nan, dtype=float),
            "snr": np.full(shape, np.nan, dtype=float),
        }

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        return MuxSingleShotGEProgram(
            self.soccfg,
            reps=1,
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
        return self.freq_offsets

    def run(self, shots=None, sweep_para=None, plot=True):
        """Run the operation.

        Parameters
        ----------
        shots : Any, default: None
            Value for ``shots``.
        sweep_para : Any, default: None
            Value for ``sweep_para``.
        plot : Any, default: True
            Value for ``plot``.

        Returns
        -------
        Any
            Result of the operation.
        """
        cfg = dict(self.cfg)
        if shots is None:
            shots = cfg["shots"]
        sweep_para = sweep_para or {}

        active_slots = list(cfg["active_slots"])
        active_ro_chs = list(cfg["active_ro_chs"])
        qubit_names = list(cfg["qubit_names"])
        trace_count = len(active_ro_chs)

        self.length_axis = self._axis(
            sweep_para.get("length", sweep_para.get("ro_length")),
            cfg["ro_length"],
        )
        self.gain_axis = self._axis(
            sweep_para.get("gain", sweep_para.get("res_gain_ge")),
            np.mean([cfg["res_gains"][slot] for slot in active_slots]),
        )
        self.freq_offsets = self._axis(
            sweep_para.get("freq_offset", sweep_para.get("freq_offsets")),
            0.0,
        )

        self.metric_arrays = self._result_arrays(
            trace_count,
            len(self.length_axis),
            len(self.gain_axis),
            len(self.freq_offsets),
        )
        raw_shape = (
            trace_count,
            len(self.length_axis),
            len(self.gain_axis),
            len(self.freq_offsets),
            2,
            int(shots),
        )
        self.iqdata = np.full(raw_shape, np.nan + 1j * np.nan, dtype=complex)

        interrupted = False
        points_done = 0
        total_points = (
            len(self.length_axis) * len(self.gain_axis) * len(self.freq_offsets)
        )
        try:
            with tqdm(total=total_points, desc="SingleShot optimize") as pbar:
                for l_idx, length in enumerate(self.length_axis):
                    for g_idx, gain in enumerate(self.gain_axis):
                        for f_idx, freq_offset in enumerate(self.freq_offsets):
                            pbar.set_postfix(
                                length=float(length),
                                gain=float(gain),
                                freq_offset=float(freq_offset),
                            )
                            point_cfg = self._point_cfg(
                                cfg,
                                active_slots,
                                float(length),
                                float(gain),
                                float(freq_offset),
                                int(shots),
                            )
                            prog = MuxSingleShotGEProgram(
                                self.soccfg,
                                reps=1,
                                final_delay=point_cfg["relax_delay"],
                                cfg=point_cfg,
                            )
                            self._last_prog = prog
                            iq_list = prog.acquire(self.soc, rounds=1, progress=False)
                            point_iq = MuxSingleShotGE._extract_iq(
                                iq_list, trace_count
                            )
                            self.iqdata[:, l_idx, g_idx, f_idx, :, :] = point_iq

                            for q_idx in range(trace_count):
                                stats = MuxSingleShotGE._analyze_ge(
                                    point_iq[q_idx, 0],
                                    point_iq[q_idx, 1],
                                )
                                self.metric_arrays["fidelity"][
                                    q_idx, l_idx, g_idx, f_idx
                                ] = stats["fidelity"]
                                self.metric_arrays["threshold"][
                                    q_idx, l_idx, g_idx, f_idx
                                ] = stats["threshold"]
                                self.metric_arrays["rotation_deg"][
                                    q_idx, l_idx, g_idx, f_idx
                                ] = stats["theta_deg"]
                                self.metric_arrays["snr"][
                                    q_idx, l_idx, g_idx, f_idx
                                ] = stats["snr"]
                            points_done += 1
                            pbar.update(1)
        except KeyboardInterrupt:
            interrupted = True

        fit_result = {}
        best = {}
        for q_idx, name in enumerate(qubit_names):
            fid_arr = self.metric_arrays["fidelity"][q_idx]
            if not np.isfinite(fid_arr).any():
                continue
            best_idx = np.unravel_index(np.nanargmax(fid_arr), fid_arr.shape)
            l_idx, g_idx, f_idx = best_idx
            best_length = float(self.length_axis[l_idx])
            best_gain = float(self.gain_axis[g_idx])
            best_freq_offset = float(self.freq_offsets[f_idx])
            best_fid = float(fid_arr[best_idx])
            best_threshold = float(self.metric_arrays["threshold"][q_idx][best_idx])
            best_rotation = float(self.metric_arrays["rotation_deg"][q_idx][best_idx])
            best_snr = float(self.metric_arrays["snr"][q_idx][best_idx])
            center_if = float(cfg["res_freqs"][active_slots[q_idx]])
            lo_ext = float(cfg.get("LO_ext") or 0.0)
            best_freq_mhz = center_if + lo_ext + best_freq_offset

            best[name] = {
                "fidelity": best_fid,
                "length": best_length,
                "gain": best_gain,
                "freq_offset_mhz": best_freq_offset,
                "res_freq_ge_mhz": best_freq_mhz,
                "threshold": best_threshold,
                "rotation_deg": best_rotation,
                "snr": best_snr,
            }
            fit_result[f"{name}_best_fidelity"] = (round(best_fid, 6), None)
            fit_result[f"{name}_best_length"] = (round(best_length, 6), None)
            fit_result[f"{name}_best_gain"] = (round(best_gain, 6), None)
            fit_result[f"{name}_best_res_freq_mhz"] = (round(best_freq_mhz, 6), None)
            fit_result[f"{name}_threshold"] = (round(best_threshold, 6), None)
            fit_result[f"{name}_rotation_deg"] = (round(best_rotation, 6), None)
            fit_result[f"{name}_snr"] = (round(best_snr, 6), None)

        self.best = best

        figures = []
        if plot and self.metric_arrays is not None:
            axes_info = [
                ("Length", self.length_axis, 0),
                ("Gain", self.gain_axis, 1),
                ("Frequency offset (MHz)", self.freq_offsets, 2),
            ]
            for q_idx, name in enumerate(qubit_names):
                fid_arr = self.metric_arrays["fidelity"][q_idx]
                fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
                varying = [info for info in axes_info if len(info[1]) > 1]
                if len(varying) >= 2:
                    x_label, x_axis, x_dim = varying[-1]
                    y_label, y_axis, y_dim = varying[-2]
                    reduce_axes = tuple(
                        dim for _, _, dim in axes_info if dim not in {x_dim, y_dim}
                    )
                    heatmap = (
                        np.nanmax(fid_arr, axis=reduce_axes)
                        if reduce_axes
                        else fid_arr
                    )
                    if (y_dim, x_dim) != tuple(sorted((y_dim, x_dim))):
                        heatmap = np.swapaxes(heatmap, 0, 1)
                    mesh = ax.pcolormesh(
                        x_axis,
                        y_axis,
                        np.asarray(heatmap),
                        shading="auto",
                    )
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    fig.colorbar(mesh, ax=ax, label="Fidelity")
                elif len(varying) == 1:
                    x_label, x_axis, x_dim = varying[0]
                    reduce_axes = tuple(
                        dim for _, _, dim in axes_info if dim != x_dim
                    )
                    line = np.nanmax(fid_arr, axis=reduce_axes)
                    ax.plot(x_axis, np.asarray(line).reshape(-1), "o-")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Fidelity")
                    ax.set_ylim(0, 1.02)
                else:
                    ax.bar([0], [float(np.asarray(fid_arr).reshape(-1)[0])])
                    ax.set_xticks([0])
                    ax.set_xticklabels(["single point"])
                    ax.set_ylabel("Fidelity")
                    ax.set_ylim(0, 1.02)
                ax.set_title(f"{name} single-shot fidelity")
                fig.tight_layout()
                figures.append(fig)
                plt.show()

        has_data = self.metric_arrays is not None and np.isfinite(self.metric_arrays["fidelity"]).any()
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self.freq_offsets,
            y_axis=self.gain_axis,
            fit_result=fit_result,
            metadata={
                "qubit_names": qubit_names,
                "active_ro_chs": active_ro_chs,
                "active_slots": active_slots,
                "length_axis": self.length_axis.tolist(),
                "gain_axis": self.gain_axis.tolist(),
                "freq_offsets_mhz": self.freq_offsets.tolist(),
                "best": best,
                "points_acquired": points_done,
                "shots": int(shots),
            },
            axes={
                "qubit": {"values": qubit_names},
                "length": {"values": self.length_axis, "unit": "us"},
                "gain": {"values": self.gain_axis, "unit": "DAC unit"},
                "frequency_offset": {"values": self.freq_offsets, "unit": "MHz"},
                "state": {"values": ["g", "e"]},
                "shot": {"values": np.arange(int(shots)), "unit": "#"},
            },
            dataset_dims={
                "iq": ["qubit", "length", "gain", "frequency_offset", "state", "shot"]
            },
            analysis_data={
                name: {
                    "values": values,
                    "dims": ["qubit", "length", "gain", "frequency_offset"],
                }
                for name, values in self.metric_arrays.items()
            },
            data_kind="single_shot_optimization",
            analysis_id="single_shot_optimization",
            plot_id="single_shot_optimization",
            figures=figures,
            quality=QualityFlag.GOOD if has_data else QualityFlag.BAD,
            quality_message="Mux single-shot ge optimization acquired."
            if has_data
            else "No data acquired.",
            interrupted=interrupted,
            avg_count=1,
        )
        self.result = result
        return result


__all__ = [
    "MuxSingleShotGE",
    "MuxSingleShotGEProgram",
    "MuxSingleShotGEOpt",
]
