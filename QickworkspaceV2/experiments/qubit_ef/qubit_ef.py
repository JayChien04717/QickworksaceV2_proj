"""
QubitEF/qubit_ef — s010-s011: EF transition spectroscopy, Rabi, and temperature.
"""

from __future__ import annotations

import numpy as np

from ...core.base_program import BaseProgram
from ...core.base_experiment import BaseExperiment
from ...core.experiment_data import ExperimentData, QualityFlag
from ...analysis.qubit import PowerRabiAnalysis, QubitTempAnalysis
from ...analysis.resonator import LorentzianAnalysis


# ── s010 — Qubit Spec EF ──────────────────────────────────────────────────────

class QubitSpecEfProgram(BaseProgram):
    """EF spectroscopy: ge π pulse then sweeps ef drive frequency."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("freqloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse", pulse_type="flat_top")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.02)
        self.measure(cfg)


class QubitSpecEf(BaseExperiment):
    """Qubit spectroscopy (ef): ge π then sweep ef frequency."""

    EXPT_NAME = "s010_qubit_spec_ef"
    TAG = "TwoTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Qubit ef Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["qb_freq_ef"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = LorentzianAnalysis

    def _create_program(self):
        return QubitSpecEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_pulse", "freq", as_array=True)


# ── s011 — Power Rabi EF ──────────────────────────────────────────────────────

class PowerRabiEfProgram(BaseProgram):
    """EF power Rabi: ge π pulse then sweep ef drive gain."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.setup_qubit_gen(cfg, "ef")
        self.add_loop("gainloop", cfg["steps"])
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")
        self.setup_qb_pulse(cfg, "ef", name="qb_ef_pulse")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("cooling", False):
            self.apply_cool(cfg)
            self.cooling_body(cfg)
        self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.05, tag="waiting")
        self.measure(cfg)


class PowerRabiEf(BaseExperiment):
    """EF power Rabi (s011): sweep ef gain → π_ef and π/2_ef gains."""

    EXPT_NAME = "s011_power_rabi_ef"
    TAG = "Rabi"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Power Rabi ef"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ef"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    Analysis = PowerRabiAnalysis

    def _create_program(self):
        return PowerRabiEfProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_pulse", "gain", as_array=True)


# ── s013 — Qubit Temperature ──────────────────────────────────────────────────

class QubitTempProgram(BaseProgram):
    """Qubit temperature: acquire shots for ground and excited states."""

    def _initialize(self, cfg):
        self.setup_resonator(cfg)
        self.setup_qubit_gen(cfg, "ge")
        self.add_loop("shotloop", cfg.get("shots", 1000))
        self.setup_qb_pulse(cfg, "ge", name="qb_ge_pi", gain_key="pi_gain_ge")

    def _body(self, cfg):
        self.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)
        if cfg.get("apply_ge_pi", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.01, tag="wait")
        self.measure(cfg)


class QubitTemp(BaseExperiment):
    """Qubit temperature measurement (s013_qubit_temp) via population ratio."""

    EXPT_NAME = "s013_qubit_temp"
    TAG = "Temperature"
    X_LABEL = "State"
    TITLE_PREFIX = "Qubit Temperature"
    X_SAVE_NAME = "State"
    X_SAVE_UNIT = ""
    X_SAVE_SCALE = 1.0

    Analysis = QubitTempAnalysis

    def _create_program(self):
        return QubitTempProgram(
            self.soccfg, reps=1,
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return np.array([0, 1])  # ground, excited

    @staticmethod
    def _shots_from_acquire(iq_list) -> np.ndarray:
        """Return one complex shot vector from QICK acquire output."""
        arr = np.asarray(iq_list[0])
        if arr.ndim >= 3 and arr.shape[-1] == 2:
            return arr[0, :, 0] + 1j * arr[0, :, 1]
        if arr.ndim >= 2 and arr.shape[-1] == 2:
            return arr[:, 0] + 1j * arr[:, 1]
        return np.asarray(iq_list[0][0]).dot([1, 1j]).reshape(-1)

    @staticmethod
    def _population_from_shots(no_pi: np.ndarray, ge_pi: np.ndarray):
        """Estimate thermal |e> population from no-pi shots."""
        g_mean = np.mean(no_pi)
        e_mean = np.mean(ge_pi)
        theta = -np.angle(e_mean - g_mean) if abs(e_mean - g_mean) > 0 else 0.0

        rot = np.exp(1j * theta)
        no_proj = np.real(no_pi * rot)
        pi_proj = np.real(ge_pi * rot)
        g_center = float(np.mean(no_proj))
        e_center = float(np.mean(pi_proj))
        threshold = 0.5 * (g_center + e_center)

        if e_center >= g_center:
            no_excited = no_proj > threshold
            pi_excited = pi_proj > threshold
        else:
            no_excited = no_proj < threshold
            pi_excited = pi_proj < threshold

        return {
            "n_excited": float(np.mean(no_excited)),
            "p_excited_after_pi": float(np.mean(pi_excited)),
            "threshold": float(threshold),
            "rotation_deg": float(np.degrees(theta)),
            "mean_no_pi": g_mean,
            "mean_ge_pi": e_mean,
        }

    def run(self, py_avg: int = 1, shots: int | None = None, iq_process: str = "abs",
            show_final_plot: bool = False, **kwargs) -> ExperimentData:
        """Acquire no-pi and ge-pi shot clouds, then estimate temperature."""
        from tqdm.auto import tqdm

        shots = int(shots or kwargs.get("SHOTS") or self.cfg.get("shots") or self.cfg.get("steps", 1000))
        self.cfg["shots"] = shots
        repeats = max(int(py_avg), 1)

        raw = {}
        for label, apply_ge_pi in [("no_pi", False), ("ge_pi", True)]:
            cfg = dict(self.cfg)
            cfg["apply_ge_pi"] = apply_ge_pi
            prog = QubitTempProgram(
                self.soccfg, reps=1,
                final_delay=cfg["relax_delay"], cfg=cfg,
            )
            chunks = []
            for _ in tqdm(range(repeats), desc=f"QubitTemp {label}"):
                iq_list = prog.acquire(self.soc, rounds=1, progress=False)
                chunks.append(self._shots_from_acquire(iq_list))
            raw[label] = np.concatenate(chunks)

        self.data = {
            "Ig": np.real(raw["no_pi"]),
            "Qg": np.imag(raw["no_pi"]),
            "Ie": np.real(raw["ge_pi"]),
            "Qe": np.imag(raw["ge_pi"]),
        }
        self.iqdata = np.vstack([raw["no_pi"], raw["ge_pi"]])
        self._sweep_vals_x = np.arange(self.iqdata.shape[1], dtype=float)
        self._sweep_vals_y = np.array([0.0, 1.0])

        pop = self._population_from_shots(raw["no_pi"], raw["ge_pi"])
        n_excited = pop["n_excited"]
        self.fit_params = np.array([n_excited])
        self.fit_errors = None

        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self._sweep_vals_x,
            y_axis=self._sweep_vals_y,
            fit_params=self.fit_params,
            fit_errors=self.fit_errors,
            fit_result={
                "n_excited": (n_excited, None),
                "p_excited_after_pi": (pop["p_excited_after_pi"], None),
                "threshold": (pop["threshold"], None),
                "rotation_deg": (pop["rotation_deg"], None),
            },
            quality=QualityFlag.NO_INFORMATION,
            quality_message="Temperature estimated from no-pi shots classified against ge-pi shots.",
            config=dict(self.cfg) if hasattr(self.cfg, "__iter__") else {},
            metadata={
                "state_labels": ["no_ge_pi", "with_ge_pi"],
                "shots_per_round": shots,
                "rounds": repeats,
                "mean_no_pi": {
                    "real": float(np.real(pop["mean_no_pi"])),
                    "imag": float(np.imag(pop["mean_no_pi"])),
                },
                "mean_ge_pi": {
                    "real": float(np.real(pop["mean_ge_pi"])),
                    "imag": float(np.imag(pop["mean_ge_pi"])),
                },
            },
            interrupted=False,
            avg_count=repeats,
            x_name="# shot",
            x_unit="#",
            x_scale=1.0,
            y_name="State",
            y_unit="",
            y_scale=1.0,
        )

        if self.Analysis is not None:
            result = self.Analysis().run(result)

        self.result = result
        return result

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """Save no-pi and ge-pi raw shots as a two-state Labber log."""
        from ...config.system_cfg import DATA_PATH
        from ...tools.system_tool import (
            config_to_yaml,
            get_next_filename_labber,
            hdf5_generator,
        )

        if self.iqdata is None:
            raise RuntimeError("Call run() first.")

        expt_name = f"{self.EXPT_NAME}_{qb_idx}" if title is None else f"{self.EXPT_NAME}_{qb_idx}_{title}"
        save_dir = BaseExperiment._data_path or DATA_PATH
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = config_all.to_yaml(q_id=qb_idx) if config_all is not None else config_to_yaml(self.cfg)

        hdf5_generator(
            filepath=file_path,
            x_info={"name": "# shot", "unit": "#", "values": self._sweep_vals_x},
            y_info={"name": "State", "unit": "", "values": self._sweep_vals_y},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=f"{dict_val}\nState 0: no ge pi\nState 1: with ge pi",
            tag=self.TAG,
        )
        print(f"Data saved to {file_path}")
