"""
EF Rabi-family experiments: power Rabi and qubit temperature.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

from ...analysis.qubit import PowerRabiAnalysis
from ...core.base_experiment import BaseExperiment
from ...core.base_program import BaseProgram
from ...core.experiment_data import ExperimentData, QualityFlag
from ...tools.fitting import decaysin, fitdecaysin


class PowerRabiEfProgram(BaseProgram):
    """EF power Rabi: ge pi pulse then sweep ef drive gain."""

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
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.02)
        self.measure(cfg)


class PowerRabiEf(BaseExperiment):
    """EF power Rabi (s011): sweep ef gain and fit pi / pi2 gains."""

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


_H = 6.62607015e-34
_KB = 1.380649e-23


class QubitTempProgram(BaseProgram):
    """EF Rabi program used for qubit temperature measurement."""

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
        if cfg.get("temp_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
        self.delay_auto(0.02)
        self.pulse(ch=cfg["qb_ch_ef"], name="qb_ef_pulse", t=0)
        self.delay_auto(0.02)
        if cfg.get("ge_ref", False):
            self.pulse(ch=cfg["qb_ch"], name="qb_ge_pi", t=0)
            self.delay_auto(0.02)
        self.measure(cfg)


class QubitTemp(BaseExperiment):
    """Qubit temperature from the EF Rabi amplitude ratio."""

    EXPT_NAME = "s013b_qubit_temp_ef"
    TAG = "Temperature"
    X_LABEL = "Dac Gain (a.u)"
    TITLE_PREFIX = "Qubit Temperature (Rabi ef)"
    SWEEP_KEYS_TO_REMOVE = ["qb_gain_ef"]
    X_SAVE_NAME = "Gain"
    X_SAVE_UNIT = "DAC unit"
    X_SAVE_SCALE = 1.0

    Analysis = None

    def _create_program(self):
        return QubitTempProgram(
            self.soccfg, reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"], cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        return prog.get_pulse_param("qb_ef_pulse", "gain", as_array=True)

    def run(
        self,
        py_avg: int,
        full_model: bool = False,
        iq_process: str | None = None,
        show_final_plot: bool = False,
        **kwargs,
    ) -> ExperimentData:
        """Run no-ge-pi and with-ge-pi EF Rabi traces, then fit temperature."""
        self._full_model = full_model

        print("[Temp] Meas run: ef Rabi only...")
        self.cfg["temp_ref"] = False
        super().run(py_avg, iq_process=iq_process, show_final_plot=show_final_plot, **kwargs)
        self._iq_meas = np.asarray(self.iqdata).copy()
        x_vals = np.asarray(self._sweep_vals_x, dtype=float)

        print("[Temp] Ref run: ge pi + ef Rabi...")
        self.cfg["temp_ref"] = True
        super().run(py_avg, iq_process=iq_process, show_final_plot=show_final_plot, **kwargs)
        self._iq_ref = np.asarray(self.iqdata).copy()
        self.cfg["temp_ref"] = False

        temp = self._compute_temperature(x_vals)
        self.iqdata = np.vstack([self._iq_meas, self._iq_ref])
        self._sweep_vals_x = x_vals
        self._sweep_vals_y = np.array([0.0, 1.0])

        fit_result = {}
        scalar = None
        fit_params = None
        quality = QualityFlag.BAD
        quality_message = "Temperature calculation failed."
        if temp is not None:
            scalar = float(temp * 1e3)
            temp_err_mk = self._T_err_K * 1e3 if self._T_err_K is not None else None
            fit_params = np.array([scalar])
            fit_result = {
                "T_mK": (scalar, temp_err_mk),
                "T_mK_err_meas_only": (
                    self._T_err_meas_only_K * 1e3
                    if self._T_err_meas_only_K is not None else None,
                    None,
                ),
                "T_floor_mK": (
                    self._T_floor_K * 1e3 if self._T_floor_K is not None else None,
                    None,
                ),
                "A_meas": (self._A_meas, self._A_meas_err),
                "A_ref": (self._A_ref, self._A_ref_err),
                "rabi_ratio": (self._ratio, self._ratio_err),
            }
            quality = QualityFlag.GOOD
            quality_message = ""

        self.fit_params = fit_params
        self.fit_errors = None
        result = ExperimentData(
            experiment_type=self.EXPT_NAME,
            raw_iq=self.iqdata,
            x_axis=self._sweep_vals_x,
            y_axis=self._sweep_vals_y,
            fit_params=fit_params,
            fit_errors=None,
            fit_result=fit_result,
            scalar_result=scalar,
            quality=quality,
            quality_message=quality_message,
            metadata={
                "state_labels": ["meas_ef_only", "ref_ge_pi_ef"],
                "full_model": full_model,
            },
            interrupted=False,
            avg_count=py_avg,
            x_name=self.X_SAVE_NAME,
            x_unit=self.X_SAVE_UNIT,
            x_scale=self.X_SAVE_SCALE,
            y_name="Meas_Type",
            y_unit="0:Meas, 1:Ref",
            y_scale=1.0,
        )
        self.result = result
        return result

    def _compute_temperature(self, x_vals):
        """Fit both EF Rabi traces and solve temperature from amplitude ratio."""
        mag_meas = np.abs(self._iq_meas)
        mag_ref = np.abs(self._iq_ref)

        try:
            p_meas, cov_meas, _ = fitdecaysin(x_vals, mag_meas)
            p_ref, cov_ref, _ = fitdecaysin(x_vals, mag_ref)
        except Exception as exc:
            print(f"[Temp] Fit failed: {exc}")
            return None

        fit_meas = decaysin(x_vals, *p_meas)
        fit_ref = decaysin(x_vals, *p_ref)
        self._A_meas = self._fit_amplitude(x_vals, p_meas)
        self._A_ref = self._fit_amplitude(x_vals, p_ref)
        self._A_meas_err = self._fit_amplitude_error(x_vals, p_meas, cov_meas)
        self._A_ref_err = self._fit_amplitude_error(x_vals, p_ref, cov_ref)
        self._ratio = self._A_meas / self._A_ref if self._A_ref != 0 else np.nan
        self._ratio_err_meas_only = self._A_meas_err / self._A_ref if self._A_ref != 0 and self._A_meas_err is not None else None
        self._ratio_err = self._ratio_error()

        print(
            f"[Temp] A_meas={self._A_meas:.4f}  "
            f"A_ref={self._A_ref:.4f}  ratio={self._ratio:.4f}"
        )

        if self._A_meas <= 0 or self._A_ref <= 0:
            print("[Temp] ERROR: non-positive amplitude; check data quality.")
            return None

        temp = self._solve_temperature(
            self._A_meas,
            self._A_ref,
            self.cfg["qb_freq_ge"] * 1e6,
            self.cfg["qb_freq_ef"] * 1e6,
        )
        if temp is not None:
            self._last_T_K = temp
            self._T_err_K = self._temperature_error(self._ratio, self._ratio_err)
            self._T_err_meas_only_K = self._temperature_error(
                self._ratio, self._ratio_err_meas_only
            )
            self._T_floor_K = self._temperature_floor()
            print(f"[Temp] Estimated temperature: {temp * 1e3:.2f} mK")
            if self._T_err_K is not None:
                print(f"[Temp] Temperature error: +/- {self._T_err_K * 1e3:.2f} mK")
            if self._T_floor_K is not None:
                print(f"[Temp] 1-sigma temperature floor: {self._T_floor_K * 1e3:.2f} mK")
            self._plot_temperature(
                x_vals, mag_meas, fit_meas, mag_ref, fit_ref,
                self._A_meas, self._A_ref, temp,
            )
        else:
            print("[Temp] Temperature calculation failed.")
        return temp

    def _fit_amplitude(self, x_vals, params):
        fit = decaysin(x_vals, *params)
        return float((np.max(fit) - np.min(fit)) / 2)

    def _fit_amplitude_error(self, x_vals, params, covariance):
        if covariance is None:
            return None
        cov = np.asarray(covariance, dtype=float)
        if cov.shape[0] != len(params) or not np.all(np.isfinite(cov)):
            return None

        params = np.asarray(params, dtype=float)
        grad = np.zeros_like(params)
        for idx, value in enumerate(params):
            step = 1e-6 * max(abs(value), 1.0)
            p_hi = params.copy()
            p_lo = params.copy()
            p_hi[idx] += step
            p_lo[idx] -= step
            grad[idx] = (
                self._fit_amplitude(x_vals, p_hi)
                - self._fit_amplitude(x_vals, p_lo)
            ) / (2 * step)

        variance = float(grad @ cov @ grad)
        if variance < 0 or not np.isfinite(variance):
            return None
        return float(np.sqrt(variance))

    def _ratio_error(self):
        if self._A_ref == 0:
            return None
        terms = []
        if self._A_meas_err is not None and self._A_meas > 0:
            terms.append((self._A_meas_err / self._A_meas) ** 2)
        if self._A_ref_err is not None and self._A_ref > 0:
            terms.append((self._A_ref_err / self._A_ref) ** 2)
        if not terms:
            return None
        return float(abs(self._ratio) * np.sqrt(sum(terms)))

    def _temperature_error(self, ratio, ratio_err):
        if ratio_err is None or ratio <= 0:
            return None
        if not self._full_model:
            log_ratio = np.log(ratio)
            if log_ratio == 0:
                return None
            fge_hz = self.cfg["qb_freq_ge"] * 1e6
            dT_dr = (_H * fge_hz) / (_KB * ratio * log_ratio**2)
            return float(abs(dT_dr) * ratio_err)

        step = max(abs(ratio) * 1e-4, 1e-8)
        t_hi = self._solve_temperature_from_ratio(ratio + step)
        t_lo = self._solve_temperature_from_ratio(max(ratio - step, 1e-12))
        if t_hi is None or t_lo is None:
            return None
        return float(abs((t_hi - t_lo) / (2 * step)) * ratio_err)

    def _temperature_floor(self):
        if self._ratio_err_meas_only is None:
            return None
        sigma = float(self.cfg.get("temp_floor_sigma", 1.0))
        floor_ratio = sigma * self._ratio_err_meas_only
        return self._solve_temperature_from_ratio(floor_ratio)

    def _solve_temperature_from_ratio(self, measured_ratio):
        if measured_ratio <= 0 or measured_ratio >= 1:
            return None
        fge_hz = self.cfg["qb_freq_ge"] * 1e6
        fef_hz = self.cfg["qb_freq_ef"] * 1e6
        if not self._full_model:
            return -(_H * fge_hz) / (_KB * np.log(measured_ratio))
        return self._solve_temperature_from_ratio_full(measured_ratio, fge_hz, fef_hz)

    def _solve_temperature(self, A_meas, A_ref, fge_hz, fef_hz):
        measured_ratio = A_meas / A_ref
        if not self._full_model:
            if measured_ratio <= 0 or measured_ratio >= 1:
                print(f"[Temp] Unphysical ratio: {measured_ratio:.4f}")
                return None
            return -(_H * fge_hz) / (_KB * np.log(measured_ratio))

        return self._solve_temperature_from_ratio_full(measured_ratio, fge_hz, fef_hz)

    def _solve_temperature_from_ratio_full(self, measured_ratio, fge_hz, fef_hz):
        if measured_ratio <= 0 or measured_ratio >= 1:
            return None
        e_ge = _H * fge_hz
        e_gf = _H * (fge_hz + fef_hz)

        def residual(temp):
            beta = 1.0 / (_KB * temp)
            p_e_raw = np.exp(-e_ge * beta)
            p_f_raw = np.exp(-e_gf * beta)
            z = 1 + p_e_raw + p_f_raw
            p_g, p_e, p_f = 1 / z, p_e_raw / z, p_f_raw / z
            return (p_e - p_f) / (p_g - p_f) - measured_ratio

        try:
            sol = root_scalar(residual, bracket=[0.005, 1.0], method="brentq")
            return sol.root if sol.converged else None
        except ValueError as exc:
            print(f"[Temp] Solver failed: {exc}")
            return None

    def _plot_temperature(self, x_vals, mag_meas, fit_meas, mag_ref, fit_ref,
                          a_meas, a_ref, temp):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, mag_meas, "o-", color="C0", markersize=5, alpha=0.7, label="Meas data")
        ax.plot(x_vals, fit_meas, color="C0", linewidth=2, label=f"Meas fit (A={a_meas:.4f})")
        ax.plot(x_vals, mag_ref, "o-", color="C1", markersize=5, alpha=0.7, label="Ref data")
        ax.plot(x_vals, fit_ref, color="C1", linewidth=2, label=f"Ref fit (A={a_ref:.4f})")
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel("Magnitude (a.u.)")
        temp_text = f"T = {temp * 1e3:.2f} mK"
        if self._T_err_K is not None:
            temp_text += f" +/- {self._T_err_K * 1e3:.2f} mK"
        if self._T_floor_K is not None:
            temp_text += f" | floor = {self._T_floor_K * 1e3:.2f} mK"
        ax.set_title(
            f"{self.TITLE_PREFIX}\n"
            f"A_meas={a_meas:.4f}, A_ref={a_ref:.4f}, ratio={a_meas / a_ref:.4f}\n"
            f"{temp_text}"
        )
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        plt.show()

    def _post_fit(self, x_vals):
        return None

    def _save_comment(self, dict_val: str) -> str:
        base_comment = super()._save_comment(dict_val)
        temp = getattr(self, "_last_T_K", None)
        if temp is not None:
            lines = [f"Calculated Qubit Temperature: {temp * 1e3:.2f} mK"]
            if self._T_err_K is not None:
                lines.append(f"Temperature Error: +/- {self._T_err_K * 1e3:.2f} mK")
            if self._T_floor_K is not None:
                lines.append(f"1-sigma Temperature Floor: {self._T_floor_K * 1e3:.2f} mK")
            return "\n".join(lines) + f"\n\n{base_comment}"
        return base_comment

    def saveLabber(self, qb_idx, yoko_value=None, config_all=None, title=None):
        """Save meas/ref EF Rabi raw data as a two-state Labber log."""
        from ...tools.system_tool import (
            config_to_yaml,
            get_next_filename_labber,
            hdf5_generator,
        )

        if self.iqdata is None:
            raise RuntimeError("Call run() first.")

        expt_name = f"{self.EXPT_NAME}_{qb_idx}" if title is None else f"{self.EXPT_NAME}_{qb_idx}_{title}"
        save_dir = BaseExperiment._require_data_path()
        file_path = get_next_filename_labber(save_dir, expt_name, yoko_value)
        dict_val = (
            config_all.to_yaml(q_id=qb_idx)
            if config_all is not None
            else config_to_yaml(self.cfg)
        )

        hdf5_generator(
            filepath=file_path,
            x_info={"name": self.X_SAVE_NAME, "unit": self.X_SAVE_UNIT, "values": self._sweep_vals_x},
            y_info={"name": "Meas_Type", "unit": "0:Meas, 1:Ref", "values": self._sweep_vals_y},
            z_info={"name": "Signal", "unit": "ADC unit", "values": self.iqdata},
            comment=self._save_comment(f"{dict_val}\nState 0: ef only\nState 1: ge pi + ef"),
            tag=self.TAG,
        )
        print(f"Data saved to {file_path}")


__all__ = ["PowerRabiEfProgram", "PowerRabiEf", "QubitTempProgram", "QubitTemp"]
