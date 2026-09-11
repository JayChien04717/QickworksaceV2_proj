"""
AutoCalibrate: Rebuilt ge calibration pipeline using the composite framework.

Reproduces the full qick_workspace auto_calibrate.py logic with:
  - ExperimentData returns from every step
  - CalibrationStore updates after each step
  - GP-guided Ramsey frequency correction preserved
  - Sequential pipeline interface
"""

from __future__ import annotations

import numpy as np

from ..core.experiment_data import ExperimentData, QualityFlag
from .store import CalibrationStore


class AutoCalibrate:
    """
    Automated single-qubit ge calibration pipeline.

    Parameters
    ----------
    config_all : ExperimentConfig
        Live config store (``QickworkspaceV2.config.system_cfg.ExperimentConfig``).
    qubit : str
        Qubit label, e.g. ``"Q1"``.
    cal_store : CalibrationStore or None
        If provided, results are persisted after each step.
    """

    TARGET_PI_GAIN = 0.5
    MAX_RABI_RETRIES = 3
    MAX_RAMSEY_RETRIES = 3
    RAMSEY_DETUNE_LIMIT_MHZ = 0.05

    def __init__(
        self,
        config_all,
        qubit: str,
        cal_store: CalibrationStore | None = None,
        init_guess: dict | None = None,
    ):
        """Initialize the AutoCalibrate instance.

        Parameters
        ----------
        config_all : Any
            Value for ``config_all``.
        qubit : str
            Qubit identifier.
        cal_store : CalibrationStore | None, default: None
            Value for ``cal_store``.
        init_guess : dict | None, default: None
            Value for ``init_guess``.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        from ..core.base_experiment import BaseExperiment
        if BaseExperiment._soc is None:
            raise RuntimeError("Call BaseExperiment.setup(soc, soccfg, data_path) first.")
        self.config_all = config_all
        self.qubit = qubit
        self.cal_store = cal_store
        self.init_guess = init_guess or {}
        self.results: dict = {}
        self._T2r: float | None = self._guess_float("T2_guess")
        self._T2e: float | None = None
        self._T1:  float | None = self._guess_float("T1_guess")


    def _cfg(self) -> dict:
        """Return the cfg result.

        Returns
        -------
        dict
            Result of the operation.
        """
        return self.config_all.get_qubit(self.qubit)

    def _guess_float(self, key: str, default: float | None = None) -> float | None:
        """Return the guess float result.

        Parameters
        ----------
        key : str
            Lookup key.
        default : float | None, default: None
            Value for ``default``.

        Returns
        -------
        float | None
            Result of the operation.
        """
        value = self.init_guess.get(key, default)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _range_guess(self, key: str, center: float, span: float, steps: int):
        """Return the range guess result.

        Parameters
        ----------
        key : str
            Lookup key.
        center : float
            Value for ``center``.
        span : float
            Value for ``span``.
        steps : int
            Value for ``steps``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        ValueError
            If the operation cannot be completed.
        """
        spec = self.init_guess.get(key)
        if spec is None:
            return float(center - span), float(center + span), int(steps)
        if not isinstance(spec, (tuple, list)) or len(spec) != 3:
            raise ValueError(f"init_guess[{key!r}] must be (start, stop, points_or_step)")
        start, stop, n_or_step = spec
        start = float(start)
        stop = float(stop)
        if isinstance(n_or_step, int) or float(n_or_step).is_integer() and float(n_or_step) > 1:
            n = int(n_or_step)
        else:
            step_size = abs(float(n_or_step))
            if step_size <= 0:
                raise ValueError(f"init_guess[{key!r}] step must be positive")
            n = int(round(abs(stop - start) / step_size)) + 1
        return start, stop, max(n, 2)

    @staticmethod
    def _expand_range(start: float, stop: float, factor: float):
        """Return the expand range result.

        Parameters
        ----------
        start : float
            Value for ``start``.
        stop : float
            Value for ``stop``.
        factor : float
            Value for ``factor``.

        Returns
        -------
        Any
            Result of the operation.
        """
        center = 0.5 * (start + stop)
        half_span = 0.5 * abs(stop - start) * factor
        return center - half_span, center + half_span

    def _update(self, key: str, value):
        """Update the operation.

        Parameters
        ----------
        key : str
            Lookup key.
        value : Any
            Value to apply.
        """
        self.config_all.update(key, value, q_index=self.qubit)
        if self.cal_store is not None:
            self.cal_store.set(self.qubit, key, value)

    def _log(self, step: str, msg: str):
        """Return the log result.

        Parameters
        ----------
        step : str
            Value for ``step``.
        msg : str
            Value for ``msg``.
        """
        print(f"  [{step}] {msg}")

    def _require_result(self, step: str, result: ExperimentData, required: tuple[str, ...] = ()) -> ExperimentData:
        """Return the require result result.

        Parameters
        ----------
        step : str
            Value for ``step``.
        result : ExperimentData
            Experiment result to process.
        required : tuple[str, ...], default: ()
            Value for ``required``.

        Returns
        -------
        ExperimentData
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        if result.quality == QualityFlag.BAD:
            raise RuntimeError(f"{step}: analysis failed: {result.quality_message}")
        missing = [key for key in required if result.get_param(key) is None]
        if missing:
            raise RuntimeError(
                f"{step}: missing fit result(s) {missing}; quality={result.quality.value}; "
                f"message={result.quality_message!r}"
            )
        return result

    def _require_param(
        self,
        step: str,
        result: ExperimentData,
        key: str,
        *,
        lo: float | None = None,
        hi: float | None = None,
    ) -> float:
        """Return the require param result.

        Parameters
        ----------
        step : str
            Value for ``step``.
        result : ExperimentData
            Experiment result to process.
        key : str
            Lookup key.
        lo : float | None, default: None
            Value for ``lo``.
        hi : float | None, default: None
            Value for ``hi``.

        Returns
        -------
        float
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        value = result.get_param(key)
        if value is None:
            if result.scalar_result is not None and key in {"scalar", "f0_MHz", "f_res[MHz]"}:
                value = result.scalar_result
            else:
                raise RuntimeError(f"{step}: missing parameter {key!r}")
        value = float(value)
        if not np.isfinite(value):
            raise RuntimeError(f"{step}: {key} is not finite ({value!r})")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise RuntimeError(f"{step}: {key}={value:.6g} outside [{lo}, {hi}]")
        return value

    @staticmethod
    def _axis_extremum_freq(result: ExperimentData) -> float | None:
        """Return the axis extremum freq result.

        Parameters
        ----------
        result : ExperimentData
            Experiment result to process.

        Returns
        -------
        float | None
            Result of the operation.
        """
        if result.x_axis is None or result.raw_iq is None:
            return None
        x = np.asarray(result.x_axis, dtype=float)
        y = np.abs(np.asarray(result.raw_iq))
        if x.size == 0 or y.size == 0:
            return None
        y = np.ravel(y)
        idx = int(np.nanargmin(y))
        if idx >= x.size:
            idx = int(np.nanargmax(y)) % x.size
        value = float(x[idx])
        return value if np.isfinite(value) else None


    def run(self, skip: tuple = ()):
        """Run the operation.

        Parameters
        ----------
        skip : tuple, default: ()
            Value for ``skip``.
        """
        PIPELINE = [
            ("res_spec",   self.step_res_spec),
            ("qubit_spec", self.step_qubit_spec),
            ("power_rabi", self.step_power_rabi),
            ("t1",         self.step_t1),
            ("ramsey",     self.step_ramsey),
            ("spin_echo",  self.step_spin_echo),
            ("ss_opt",     self.step_ss_opt),
        ]
        for name, fn in PIPELINE:
            if name in skip:
                print(f"[auto_cal] skip: {name}")
                continue
            print(f"\n{'='*60}")
            print(f"  AUTO-CAL  |  {name}  |  qubit = {self.qubit}")
            print(f"{'='*60}")
            fn()


    def step_res_spec(self, span=10.0, steps=101, py_avg=5):
        """Return the step res spec result.

        Parameters
        ----------
        span : Any, default: 10.0
            Value for ``span``.
        steps : Any, default: 101
            Value for ``steps``.
        py_avg : Any, default: 5
            Number of Python-level acquisition averages.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        from qick.asm_v2 import QickSweep1D
        from ..experiments.resonator.res_spec import ResonatorSpec
        center = self._cfg()["res_freq_ge"]
        start0, stop0, steps = self._range_guess("res", center, span, steps)
        for attempt, factor in enumerate([1.0, 2.0, 4.0]):
            start, stop = self._expand_range(start0, stop0, factor)
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("res_freq_ge", QickSweep1D("freqloop", start, stop)),
                ("relax_delay", 0),
            ])
            expt = ResonatorSpec(run_cfg)
            result: ExperimentData = expt.run(py_avg, solve_type="hm")
            try:
                self._require_result("res_spec", result)
                freq_mhz = round(self._require_param("res_spec", result, "f_res[MHz]", lo=100.0, hi=20000.0), 4)
                self._update("res_freq_ge", freq_mhz)
                self.results["res_freq_ge"] = freq_mhz
                self._log("res_spec", f"res_freq_ge = {freq_mhz} MHz")
                return freq_mhz
            except RuntimeError as exc:
                fallback_freq = self._axis_extremum_freq(result)
                if fallback_freq is not None:
                    freq_mhz = round(fallback_freq, 4)
                    self._update("res_freq_ge", freq_mhz)
                    self.results["res_freq_ge"] = freq_mhz
                    self._log("res_spec", f"fit failed; accepted sweep extremum fallback res_freq_ge={freq_mhz} MHz")
                    return freq_mhz
                self._log("res_spec", f"{exc} (attempt {attempt+1}), expanding range to {start:.4f}-{stop:.4f} MHz")
        raise RuntimeError("res_spec: circle-fit failed after 3 attempts")


    def step_qubit_spec(self, span=50.0, steps=101, py_avg=5):
        """Return the step qubit spec result.

        Parameters
        ----------
        span : Any, default: 50.0
            Value for ``span``.
        steps : Any, default: 101
            Value for ``steps``.
        py_avg : Any, default: 5
            Number of Python-level acquisition averages.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        from qick.asm_v2 import QickSweep1D
        from ..experiments.qubit_ge.qubit_spec import QubitSpec
        center = self._cfg()["qb_freq_ge"]
        start0, stop0, steps = self._range_guess("qb", center, span, steps)
        for attempt, factor in enumerate([1.0, 2.0, 4.0]):
            start, stop = self._expand_range(start0, stop0, factor)
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("qb_freq_ge", QickSweep1D("freqloop", start, stop)),
                ("qb_mixer", center),
                ("qb_gain_ge", 0.1),
                ("qb_flat_top_length_ge", 1.0),
                ("relax_delay", 1),
            ])
            expt = QubitSpec(run_cfg)
            result: ExperimentData = expt.run(py_avg)
            try:
                self._require_result("qubit_spec", result)
                freq_mhz = round(self._require_param("qubit_spec", result, "f0_MHz", lo=100.0, hi=20000.0), 4)
                self._update("qb_freq_ge", freq_mhz)
                self._update("qb_mixer", freq_mhz)
                self.results["qb_freq_ge"] = freq_mhz
                if result.fit_params is not None and len(result.fit_params) > 1:
                    linewidth = abs(float(result.fit_params[1]))
                    if linewidth > 1e-6:
                        self._T2r = round(1.0 / (np.pi * linewidth), 3)
                        self._log("qubit_spec", f"linewidth={linewidth:.3f} MHz → T2* seed={self._T2r:.2f} µs")
                self._log("qubit_spec", f"qb_freq_ge = {freq_mhz} MHz")
                return freq_mhz
            except RuntimeError as exc:
                self._log("qubit_spec", f"{exc} (attempt {attempt+1}), expanding range to {start:.4f}-{stop:.4f} MHz")
        raise RuntimeError("qubit_spec: Lorentzian fit failed after 3 attempts")


    def step_power_rabi(self, steps=100, py_avg=10):
        """Return the step power rabi result.

        Parameters
        ----------
        steps : Any, default: 100
            Value for ``steps``.
        py_avg : Any, default: 10
            Number of Python-level acquisition averages.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        from qick.asm_v2 import QickSweep1D
        from ..experiments.qubit_ge.rabi import PowerRabi
        rabi_start, rabi_stop, steps = self._range_guess("rabi", 0.5, 0.5, steps)
        for attempt in range(self.MAX_RABI_RETRIES):
            run_cfg = self._cfg()
            sigma = self._guess_float("sigma", run_cfg.get("sigma_ge", 0.01))
            if sigma is not None:
                self._update("sigma_ge", sigma)
            run_cfg.update([
                ("steps", steps),
                ("qb_gain_ge", QickSweep1D("gainloop", rabi_start, rabi_stop)),
            ])
            expt = PowerRabi(run_cfg)
            result: ExperimentData = expt.run(py_avg)
            self._require_result("power_rabi", result, required=("pi_gain", "pi2_gain"))
            pi_gain = self._require_param("power_rabi", result, "pi_gain", lo=0.0, hi=1.2)
            pi2_gain = self._require_param("power_rabi", result, "pi2_gain", lo=0.0, hi=1.2)
            if 0.15 <= pi_gain <= 0.90:
                self._update("pi_gain_ge",  round(pi_gain,  6))
                self._update("pi2_gain_ge", round(pi2_gain, 6))
                self.results["pi_gain_ge"]  = pi_gain
                self.results["pi2_gain_ge"] = pi2_gain
                self._log("power_rabi", f"pi_gain_ge={pi_gain:.4f}, pi2_gain_ge={pi2_gain:.4f}")
                return pi_gain, pi2_gain
            new_sigma = round(sigma * (pi_gain / self.TARGET_PI_GAIN), 5)
            direction = "increase" if pi_gain > 0.90 else "decrease"
            self._log("power_rabi", f"attempt {attempt+1}: pi_gain={pi_gain:.4f} out of range → "
                      f"{direction} sigma_ge: {sigma:.5f} → {new_sigma:.5f}")
            self._update("sigma_ge", new_sigma)
        raise RuntimeError(f"power_rabi: pi_gain still out of range after {self.MAX_RABI_RETRIES} retries")


    def step_ramsey(self, steps=100, py_avg=20, virtual_detune=2.0):
        """Return the step ramsey result.

        Parameters
        ----------
        steps : Any, default: 100
            Value for ``steps``.
        py_avg : Any, default: 20
            Number of Python-level acquisition averages.
        virtual_detune : Any, default: 2.0
            Value for ``virtual_detune``.

        Returns
        -------
        Any
            Result of the operation.

        Raises
        ------
        RuntimeError
            If the operation cannot be completed.
        """
        from qick.asm_v2 import QickSweep1D
        from ..experiments.coherence.ramsey import Ramsey
        virtual_detune = self._guess_float("virtual_detune", virtual_detune) or virtual_detune
        stop = max(2.0, min(3.0 * self._T1, 15.0)) if self._T1 else 5.0
        _, stop, steps = self._range_guess("ramsey", 0.0, stop, steps)
        _freq_history:  list[float] = []
        _error_history: list[float] = []

        for attempt in range(self.MAX_RAMSEY_RETRIES + 2):
            run_cfg = self._cfg()
            run_cfg.update([
                ("steps", steps),
                ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
                ("virtual_detune", virtual_detune),
            ])
            expt = Ramsey(run_cfg)
            result: ExperimentData = expt.run(py_avg)
            self._require_result("ramsey", result, required=("T2r_us",))
            T2r = self._require_param("ramsey", result, "T2r_us", lo=0.01, hi=5000.0)
            detune = result.get_param("detune_MHz")
            if detune is None:
                raise RuntimeError("ramsey: detune_MHz missing; use nonzero virtual_detune")
            signed_err = float(detune) - virtual_detune
            abs_err    = abs(signed_err)
            self._T2r  = T2r
            self.results["T2r_us"] = T2r
            _freq_history.append(self._cfg()["qb_freq_ge"])
            _error_history.append(signed_err)
            self._log("ramsey", f"attempt {attempt+1}: T2*={T2r:.2f} µs, detune error={signed_err:+.4f} MHz")

            if abs_err < self.RAMSEY_DETUNE_LIMIT_MHZ:
                corrected = expt.correct_detune()
                self._update("qb_freq_ge", corrected)
                self.results["qb_freq_ge_corrected"] = corrected
                self._log("ramsey", f"converged → qb_freq_ge={corrected:.4f} MHz")
                return T2r, corrected

            gp_freq = None
            if len(_freq_history) >= 2:
                gp_freq = self._gp_predict_zero_crossing(_freq_history, _error_history)
            if gp_freq is not None:
                self._update("qb_freq_ge", gp_freq)
                self._log("ramsey", f"  → GP correction: qb_freq_ge={gp_freq:.4f} MHz")
            else:
                corrected = expt.correct_detune()
                self._update("qb_freq_ge", corrected)
            stop = max(2.0, min(3.0 * T2r, 15.0))

        self._log("ramsey", f"did not converge in {self.MAX_RAMSEY_RETRIES+2} attempts, proceeding")
        return self._T2r, self._cfg()["qb_freq_ge"]


    def step_spin_echo(self, steps=100, py_avg=100):
        """Return the step spin echo result.

        Parameters
        ----------
        steps : Any, default: 100
            Value for ``steps``.
        py_avg : Any, default: 100
            Number of Python-level acquisition averages.

        Returns
        -------
        Any
            Result of the operation.
        """
        from qick.asm_v2 import QickSweep1D
        from ..experiments.coherence.spin_echo import SpinEcho
        T2r = self._T2r or min(self._T1 or self._guess_float("T2_guess", 2.0) or 2.0, 2.0)
        T2e_est = 3.0 * T2r
        stop = max(20.0, 5.0 * T2e_est)
        _, stop, steps = self._range_guess("spin_echo", 0.0, stop, steps)
        virtual_detune = round(min(0.5, 1.5 / stop), 4)
        run_cfg = self._cfg()
        run_cfg.update([
            ("steps", steps),
            ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
            ("virtual_detune", virtual_detune),
        ])
        expt = SpinEcho(run_cfg)
        result: ExperimentData = expt.run(py_avg)
        self._require_result("spin_echo", result, required=("T2e_us",))
        T2e = self._require_param("spin_echo", result, "T2e_us", lo=0.01, hi=10000.0)
        self._T2e = T2e
        self.results["T2e_us"] = T2e
        self._log("spin_echo", f"T2E={T2e:.2f} µs (stop={stop:.0f} µs, virtual_detune={virtual_detune} MHz)")
        return T2e


    def step_t1(self, steps=100, py_avg=50):
        """Return the step t1 result.

        Parameters
        ----------
        steps : Any, default: 100
            Value for ``steps``.
        py_avg : Any, default: 50
            Number of Python-level acquisition averages.

        Returns
        -------
        Any
            Result of the operation.
        """
        from qick.asm_v2 import QickSweep1D
        from ..experiments.coherence.t1 import T1
        T_ref = self._T1 or self._T2r or 5.0
        stop  = max(50.0, 3.0 * T_ref)
        _, stop, steps = self._range_guess("t1", 0.0, stop, steps)
        relax = max(float(self._cfg().get("relax_delay", 100)), 5.0 * T_ref)
        self._update("relax_delay", round(relax))
        run_cfg = self._cfg()
        run_cfg.update([
            ("steps", steps),
            ("wait_time", QickSweep1D("waitloop", 0.0, stop)),
        ])
        expt = T1(run_cfg)
        result: ExperimentData = expt.run(py_avg)
        self._require_result("t1", result, required=("T1_us",))
        T1_val = self._require_param("t1", result, "T1_us", lo=0.01, hi=10000.0)
        self._T1 = T1_val
        self.results["T1_us"] = T1_val
        new_relax = max(100, round(5.0 * T1_val))
        self._update("relax_delay", new_relax)
        self._log("t1", f"T1={T1_val:.2f} µs → relax_delay={new_relax} µs")
        return T1_val


    def step_ss_opt(self, shots=1000, coarse_pts=(5, 5)):
        """Optimize only readout length and gain with empirical shot metrics."""
        from ..experiments.setup.single_shot import SingleShot_ge_opt, SingleShot_gef

        run_cfg = self._cfg()
        n_len, n_gain = coarse_pts
        length0 = float(run_cfg["ro_length"])
        gain0 = float(run_cfg["res_gain_ge"])
        sweep_para = {
            "length": np.linspace(max(0.5, 0.6 * length0), 1.4 * length0, n_len),
            "gain": np.linspace(max(0.01, 0.6 * gain0), min(0.95, 1.4 * gain0), n_gain),
        }
        self._log("ss_opt", f"Empirical length x gain grid: {n_len}x{n_gain}")
        optimizer = SingleShot_ge_opt(run_cfg)
        optimizer.run(shots, sweep_para=sweep_para)
        length, gain = optimizer.analyze()

        self._update("ro_length", length)
        self._update("res_gain_ge", gain)
        self._update("res_phase", 0)

        run_cfg = self._cfg()
        single_shot = SingleShot_gef(run_cfg)
        single_shot.run(5000, shot_f=False)
        result = single_shot.plot(fid_avg=True, verbose=True)
        phase = float(result[2])
        fidelity = float(result[0][0]) if hasattr(result[0], "__len__") else float(result[0])
        self._update("res_phase", phase)

        self.results.update(dict(
            ro_length=length, res_gain_ge=gain,
            res_freq_ge=run_cfg["res_freq_ge"],
            res_phase_deg=phase, readout_fidelity=fidelity,
        ))
        self._log(
            "ss_opt",
            f"Best: length={length:.3f} us, gain={gain:.5f}, fidelity={fidelity:.4f}",
        )
        return length, gain

    def _gp_predict_zero_crossing(self, freq_vals: list[float], error_vals: list[float]) -> float | None:
        """Return the gp predict zero crossing result.

        Parameters
        ----------
        freq_vals : list[float]
            Value for ``freq_vals``.
        error_vals : list[float]
            Value for ``error_vals``.

        Returns
        -------
        float | None
            Result of the operation.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            return None
        try:
            X = np.array(freq_vals).reshape(-1, 1)
            y = np.array(error_vals)
            x_centre = float(X.mean())
            x_scale  = max(float(X.std()), 1e-3)
            Xn = (X - x_centre) / x_scale
            kernel = (
                RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.0))
            )
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(Xn, y)
            x_lo = freq_vals[-1] - 5.0 * x_scale
            x_hi = freq_vals[-1] + 5.0 * x_scale
            x_search = np.linspace(x_lo, x_hi, 2000)
            y_pred, y_std = gp.predict((x_search - x_centre).reshape(-1, 1) / x_scale, return_std=True)
            zero_idx   = int(np.argmin(np.abs(y_pred)))
            pred_freq  = round(float(x_search[zero_idx]), 4)
            pred_sigma = float(y_std[zero_idx])
            self._log("ramsey", f"  GP zero-crossing: {pred_freq:.4f} MHz (σ={pred_sigma:.4f} MHz)")
            if pred_sigma > 5.0 * self.RAMSEY_DETUNE_LIMIT_MHZ:
                self._log("ramsey", "  GP uncertainty too large — falling back to naive correction")
                return None
            return pred_freq
        except Exception as exc:
            self._log("ramsey", f"  GP prediction failed ({exc}) — using naive correction")
            return None


    def summary(self):
        """Return a summary of the current state."""
        print(f"\n{'='*60}")
        print(f"  Auto-Calibration Summary  —  qubit {self.qubit}")
        print(f"{'='*60}")
        for key, val in self.results.items():
            print(f"  {key:<28s} = {val:.6g}" if isinstance(val, float) else f"  {key:<28s} = {val}")
        print(f"{'='*60}")
