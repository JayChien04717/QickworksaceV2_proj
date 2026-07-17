"""
Mux single-qubit state tomography.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2
from tqdm.auto import tqdm

from ..core.base_experiment import BaseExperiment
from ..core.base_program import resolve_gate
from ..core.experiment_data import ExperimentData, QualityFlag


class MuxTomographyProgram(AveragerProgramV2):
    """One tomography axis for all armed qubits with mux readout."""

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

        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.declare_gen(ch=cfg["qb_ch"][slot], nqz=cfg["nqz_qb"][slot])

        self.add_pulse(
            ch=res_ch,
            name="mux_readout",
            style="const",
            length=cfg["res_length"],
            mask=cfg["mask"],
        )

        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self._add_standard_gates(cfg, slot, name)

    def _add_standard_gates(self, cfg, slot, name):
        gates = [
            ("x180_ge", 0, "pi_gain_ge"),
            ("y180_ge", 90, "pi_gain_ge"),
            ("x90_ge", 0, "pi2_gain_ge"),
            ("x90m_ge", 180, "pi2_gain_ge"),
            ("y90_ge", 90, "pi2_gain_ge"),
            ("y90m_ge", -90, "pi2_gain_ge"),
        ]
        for gate_name, phase, gain_key in gates:
            self._add_gate_pulse(cfg, slot, f"{name}_{gate_name}", phase, cfg[gain_key][slot])

    def _add_gate_pulse(self, cfg, slot, pulse_name, phase, gain):
        qb_ch = cfg["qb_ch"][slot]
        pulse_type = cfg["pulse_type"][slot]
        if pulse_type == "const":
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="const",
                length=cfg["qb_flat_top_length_ge"][slot],
                freq=cfg["qb_freq_ge"][slot],
                phase=phase,
                gain=gain,
            )
        elif pulse_type == "arb":
            env_name = f"{pulse_name}_env"
            self.add_gauss(
                ch=qb_ch,
                name=env_name,
                sigma=cfg["sigma_ge"][slot],
                length=5 * cfg["sigma_ge"][slot],
                even_length=True,
            )
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="arb",
                envelope=env_name,
                freq=cfg["qb_freq_ge"][slot],
                phase=phase,
                gain=gain,
            )
        else:
            env_name = f"{pulse_name}_env"
            self.add_gauss(
                ch=qb_ch,
                name=env_name,
                sigma=cfg["sigma_ge"][slot],
                length=5 * cfg["sigma_ge"][slot],
                even_length=True,
            )
            self.add_pulse(
                ch=qb_ch,
                name=pulse_name,
                style="flat_top",
                envelope=env_name,
                length=cfg["qb_flat_top_length_ge"][slot],
                freq=cfg["qb_freq_ge"][slot],
                phase=phase,
                gain=gain,
            )

    def _pulse_all(self, cfg, gate):
        resolved = resolve_gate(gate)
        if resolved in ("I", "-I", None, "None"):
            return
        for slot, name in zip(cfg["active_slots"], cfg["qubit_names"]):
            self.pulse(ch=cfg["qb_ch"][slot], name=f"{name}_{resolved}", t=0)

    def _body(self, cfg):
        cal_pulse = cfg.get("cal_pulse")
        prep_pulse = cfg.get("prep_pulse")
        axis = cfg["tomo_axis"]

        if cal_pulse not in (None, "None"):
            self._pulse_all(cfg, cal_pulse)
            self.delay_auto(0.05)
        elif prep_pulse not in (None, "None"):
            self._pulse_all(cfg, prep_pulse)
            self.delay_auto(0.05)

        if axis == "X":
            self._pulse_all(cfg, "y90m")
            self.delay_auto(0.01)
        elif axis == "Y":
            self._pulse_all(cfg, "x90")
            self.delay_auto(0.01)

        self.delay_auto(0.05)
        self.trigger(ros=cfg["active_ro_chs"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="mux_readout", t=0)


class MuxTomography(BaseExperiment):
    """Mux state tomography for all armed qubits."""

    EXPT_NAME = "s016_mux_tomography_ge"
    TAG = "MuxTomography"
    TITLE_PREFIX = "Mux Tomography"

    def __init__(self, config):
        super().__init__(config)
        self.iq_g = None
        self.iq_e = None
        self.tomo_data_raw = {}
        self.expect_values = {}
        self.rho_mle = {}
        self.prep_pulse_name = None
        self._I = np.array([[1, 0], [0, 1]], dtype=complex)
        self._sx = np.array([[0, 1], [1, 0]], dtype=complex)
        self._sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._sz = np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def _extract_iq(iq_list, n_trace):
        vals = []
        for idx in range(n_trace):
            arr = np.asarray(iq_list[idx][0]).squeeze()
            if arr.ndim > 0 and arr.shape[-1] == 2:
                vals.append(np.asarray(arr).dot([1, 1j]).reshape(-1)[0])
            else:
                vals.append(np.asarray(arr, dtype=complex).reshape(-1)[0])
        return np.asarray(vals, dtype=complex)

    def _acquire_axis(self, cfg, py_avg):
        prog = MuxTomographyProgram(
            self.soccfg,
            reps=cfg["reps"],
            final_delay=cfg["relax_delay"],
            cfg=cfg,
        )
        iq_list = prog.acquire(self.soc, rounds=py_avg, progress=False)
        return self._extract_iq(iq_list, len(cfg["active_ro_chs"]))

    def _project_to_expect(self, iq_data, iq_g, iq_e):
        cal_vector = iq_e - iq_g
        data_vector = iq_data - iq_g
        projection = np.real(data_vector * np.conj(cal_vector)) / (np.abs(cal_vector) ** 2 + 1e-12)
        return float(np.clip(1 - 2 * projection, -1, 1))

    @staticmethod
    def _mle_reconstruction(rho_raw):
        eig_vals, eig_vecs = np.linalg.eigh(rho_raw)
        eig_vals = np.maximum(0, eig_vals)
        trace = np.sum(eig_vals)
        eig_vals = eig_vals / trace if trace > 0 else eig_vals
        return eig_vecs @ np.diag(eig_vals) @ np.conj(eig_vecs.T)

    def run(self, py_avg=1, prep_pulse_name=None, plot=True):
        cfg = dict(self.cfg)
        qubit_names = list(cfg["qubit_names"])
        self.prep_pulse_name = prep_pulse_name
        interrupted = False

        try:
            g_cfg = dict(cfg)
            g_cfg.update({"tomo_axis": "Z", "cal_pulse": None, "prep_pulse": None})
            e_cfg = dict(cfg)
            e_cfg.update({"tomo_axis": "Z", "cal_pulse": "x180", "prep_pulse": None})
            self.iq_g = self._acquire_axis(g_cfg, py_avg)
            self.iq_e = self._acquire_axis(e_cfg, py_avg)

            resolved_prep = resolve_gate(prep_pulse_name) if prep_pulse_name else None
            self.tomo_data_raw = {}
            for axis in tqdm(["X", "Y", "Z"], desc="Mux Tomography"):
                run_cfg = dict(cfg)
                run_cfg.update({"tomo_axis": axis, "cal_pulse": None, "prep_pulse": resolved_prep})
                self.tomo_data_raw[axis] = self._acquire_axis(run_cfg, py_avg)
        except KeyboardInterrupt:
            interrupted = True

        fit_result = {}
        expectations = []
        purities = []
        if self.tomo_data_raw:
            for qidx, name in enumerate(qubit_names):
                ev = {
                    axis: self._project_to_expect(
                        self.tomo_data_raw[axis][qidx], self.iq_g[qidx], self.iq_e[qidx]
                    )
                    for axis in ["X", "Y", "Z"]
                }
                rho_raw = 0.5 * (
                    self._I + ev["X"] * self._sx + ev["Y"] * self._sy + ev["Z"] * self._sz
                )
                rho = self._mle_reconstruction(rho_raw)
                purity = float(np.real(np.trace(rho @ rho)))
                self.expect_values[name] = ev
                self.rho_mle[name] = rho
                expectations.append([ev["X"], ev["Y"], ev["Z"]])
                purities.append(purity)
                fit_result[f"{name}_expect_X"] = (round(ev["X"], 6), None)
                fit_result[f"{name}_expect_Y"] = (round(ev["Y"], 6), None)
                fit_result[f"{name}_expect_Z"] = (round(ev["Z"], 6), None)
                fit_result[f"{name}_purity"] = (round(purity, 6), None)

        figures = []
        if plot and self.rho_mle:
            fig, axes = plt.subplots(
                len(qubit_names), 2, figsize=(8, max(3, 2.8 * len(qubit_names))), squeeze=False
            )
            cmap = plt.get_cmap("RdBu")
            labels = ["|0>", "|1>"]
            for row, name in enumerate(qubit_names):
                for col, (part, title) in enumerate([(self.rho_mle[name].real, "Real"), (self.rho_mle[name].imag, "Imag")]):
                    vmax = max(np.max(np.abs(part)), 1e-9)
                    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
                    ax = axes[row, col]
                    im = ax.matshow(part, cmap=cmap, norm=norm)
                    ax.set_title(f"{name} {title}")
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(labels)
                    ax.set_yticklabels(labels)
                    ax.xaxis.set_ticks_position("bottom")
                    fig.colorbar(im, ax=ax, shrink=0.75)
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, f"{part[i, j]:.2f}", ha="center", va="center")
            fig.suptitle(self.TITLE_PREFIX + (" (Interrupted)" if interrupted else ""))
            fig.tight_layout()
            figures.append(fig)
            plt.show()

        y_axis = np.asarray(expectations, dtype=float) if expectations else None
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq={
                "calibration_g": {"values": self.iq_g, "dims": ["qubit"]},
                "calibration_e": {"values": self.iq_e, "dims": ["qubit"]},
                "tomography": {
                    axis: {"values": values, "dims": ["qubit"]}
                    for axis, values in self.tomo_data_raw.items()
                },
            },
            x_axis=np.array([0.0, 1.0, 2.0]),
            fit_result=fit_result,
            metadata={
                "qubit_names": qubit_names,
                "axes": ["X", "Y", "Z"],
                "prep_pulse": self.prep_pulse_name,
                "purities": purities,
            },
            axes={
                "qubit": {"values": qubit_names},
                "tomography_axis": {"values": ["X", "Y", "Z"]},
                "ket": {"values": ["0", "1"]},
                "bra": {"values": ["0", "1"]},
            },
            analysis_data={
                "expectation": {
                    "values": y_axis if y_axis is not None else np.empty((0, 3)),
                    "dims": ["qubit", "tomography_axis"],
                },
                "density_matrix": {
                    "values": np.asarray([self.rho_mle[name] for name in qubit_names])
                    if self.rho_mle else np.empty((0, 2, 2), dtype=complex),
                    "dims": ["qubit", "bra", "ket"],
                },
                "purity": {"values": np.asarray(purities), "dims": ["qubit"]},
            },
            data_kind="tomography",
            analysis_id="tomography",
            plot_id="density_matrix",
            figures=figures,
            quality=QualityFlag.GOOD if self.rho_mle else QualityFlag.BAD,
            interrupted=interrupted,
            avg_count=py_avg,
        )
        self.result = result
        return result


MuxStateTomography = MuxTomography


__all__ = ["MuxStateTomography", "MuxTomography", "MuxTomographyProgram"]
