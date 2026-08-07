"""Restartable, same-flux-normalized TWPA calibration workflow.

This is the hardware orchestration layer used by ``twpa_workflow.ipynb``.
Legacy TWPA classes remain in :mod:`.twpa` for notebook compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ...core.base_experiment import BaseExperiment
from .twpa import TWPAFluxProgram


def _axis(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty, finite 1D sequence")
    return result


@dataclass(frozen=True)
class TWPASweepPlan:
    """Complete, explicit settings for one TWPA calibration run."""

    signal_start_mhz: float
    signal_stop_mhz: float
    signal_steps: int
    flux_values: Sequence[float]
    pump_freqs_hz: Sequence[float]
    pump_powers_dbm: Sequence[float]
    resonator_gain: float = 0.1
    py_avg: int = 1
    relax_delay: float = 0.0
    pump_settle_s: float = 0.1
    yoko_mode: str = "current"
    target_f_min_hz: float | None = None
    target_f_max_hz: float | None = None
    gain_threshold_db: float = 12.0
    gain_target_db: float = 15.0
    ripple_limit_db: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "flux_values", _axis(self.flux_values, "flux_values"))
        object.__setattr__(
            self, "pump_freqs_hz", _axis(self.pump_freqs_hz, "pump_freqs_hz")
        )
        object.__setattr__(
            self,
            "pump_powers_dbm",
            _axis(self.pump_powers_dbm, "pump_powers_dbm"),
        )
        if self.signal_steps < 2 or self.signal_stop_mhz <= self.signal_start_mhz:
            raise ValueError("Signal sweep must contain at least two increasing points")
        if self.py_avg < 1 or self.pump_settle_s < 0:
            raise ValueError("py_avg must be positive and pump_settle_s non-negative")
        if self.yoko_mode not in {"current", "voltage"}:
            raise ValueError("yoko_mode must be 'current' or 'voltage'")
        f_min = (
            self.signal_start_mhz * 1e6
            if self.target_f_min_hz is None
            else self.target_f_min_hz
        )
        f_max = (
            self.signal_stop_mhz * 1e6
            if self.target_f_max_hz is None
            else self.target_f_max_hz
        )
        if f_max <= f_min:
            raise ValueError("target_f_max_hz must exceed target_f_min_hz")
        object.__setattr__(self, "target_f_min_hz", float(f_min))
        object.__setattr__(self, "target_f_max_hz", float(f_max))

    def build_run_cfg(self, base_cfg: Mapping[str, Any]) -> dict[str, Any]:
        """Build an independent QICK config; never mutate ``base_cfg``."""

        from qick.asm_v2 import QickSweep1D

        result = dict(base_cfg)
        result.update(
            steps=int(self.signal_steps),
            res_gain_ge=float(self.resonator_gain),
            res_freq_ge=QickSweep1D(
                "freqloop", self.signal_start_mhz, self.signal_stop_mhz
            ),
            relax_delay=float(self.relax_delay),
            yoko_value=self.flux_values.copy(),
        )
        return result

    def refined(
        self,
        candidate: Mapping[str, float],
        *,
        flux_half_width: float = 4e-6,
        flux_steps: int = 9,
        pump_freq_half_width_hz: float = 20e6,
        pump_freq_steps: int = 9,
        pump_power_half_width_db: float = 1.0,
        pump_power_steps: int = 5,
    ) -> "TWPASweepPlan":
        """Return a fine-scan plan centered on a ranked candidate."""

        return replace(
            self,
            flux_values=np.linspace(
                candidate["ifbl"] - flux_half_width,
                candidate["ifbl"] + flux_half_width,
                flux_steps,
            ),
            pump_freqs_hz=np.linspace(
                candidate["pump_freq"] - pump_freq_half_width_hz,
                candidate["pump_freq"] + pump_freq_half_width_hz,
                pump_freq_steps,
            ),
            pump_powers_dbm=np.linspace(
                candidate["pump_power"] - pump_power_half_width_db,
                candidate["pump_power"] + pump_power_half_width_db,
                pump_power_steps,
            ),
        )


def new_twpa_run_directory(root: str | os.PathLike[str]) -> Path:
    """Create a unique timestamped run directory."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path, suffix = Path(root) / stamp, 1
    while path.exists():
        path = Path(root) / f"{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def latest_twpa_run_directory(root: str | os.PathLike[str]) -> Path:
    """Find the newest directory containing reference.nc and scan.nc."""

    paths = [
        path
        for path in Path(root).iterdir()
        if path.is_dir()
        and (path / "reference.nc").exists()
        and (path / "scan.nc").exists()
    ]
    if not paths:
        raise FileNotFoundError(f"No complete TWPA run found below {root}")
    return max(paths, key=lambda path: path.stat().st_mtime)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as opened:
        return opened.load()


def _save(dataset: xr.Dataset, path: Path) -> Path:
    """Write a checkpoint atomically so an interrupted write is recoverable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    dataset.to_netcdf(partial)
    os.replace(partial, path)
    return path


def _s21(dataset: xr.Dataset) -> xr.DataArray:
    return dataset["s21_real"] + 1j * dataset["s21_imag"]


def _matches(actual: xr.DataArray, expected: Sequence[float]) -> bool:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return left.shape == right.shape and np.allclose(
        left, right, rtol=1e-12, atol=1e-15
    )


class TWPACalibrator:
    """Acquire reference and pump-on maps with checkpoint/resume support."""

    def __init__(
        self,
        run_cfg: Mapping[str, Any],
        plan: TWPASweepPlan,
        *,
        pump_source: Any,
        instrument_manager: Any,
        yoko_name: str,
    ) -> None:
        self.run_cfg = dict(run_cfg)
        self.plan = plan
        self.pump = pump_source
        self.instruments = instrument_manager
        self.yoko_name = yoko_name
        self.soc, self.soccfg = BaseExperiment._runtime.require_hardware()

    def _program(self) -> tuple[TWPAFluxProgram, np.ndarray]:
        program = TWPAFluxProgram(
            self.soccfg,
            reps=self.run_cfg["reps"],
            final_delay=self.run_cfg["relax_delay"],
            cfg=self.run_cfg,
        )
        frequency = np.asarray(
            program.get_pulse_param("res_pulse", "freq", as_array=True) * 1e6,
            dtype=float,
        )
        return program, frequency

    def _set_flux(self, value: float) -> None:
        self.instruments.set_value(
            self.yoko_name, float(value), mode=self.plan.yoko_mode
        )

    def _trace(self, program: TWPAFluxProgram) -> np.ndarray:
        iq = program.acquire(self.soc, rounds=self.plan.py_avg, progress=False)
        return np.asarray(iq[0][0]).dot([1, 1j])

    def _reference_template(self, frequency: np.ndarray) -> xr.Dataset:
        shape = (self.plan.flux_values.size, frequency.size)
        return xr.Dataset(
            {
                "s21_real": (("ifbl", "frequency"), np.full(shape, np.nan)),
                "s21_imag": (("ifbl", "frequency"), np.full(shape, np.nan)),
                "completed": (("ifbl",), np.zeros(shape[0], dtype=np.int8)),
            },
            coords={"ifbl": self.plan.flux_values, "frequency": frequency},
            attrs={
                "kind": "twpa_pump_off_reference",
                "created_utc": _now(),
                "pump_state": 0,
                "yoko_mode": self.plan.yoko_mode,
                "frequency_unit": "Hz",
                "flux_unit": "A" if self.plan.yoko_mode == "current" else "V",
            },
        )

    def _scan_template(self, frequency: np.ndarray) -> xr.Dataset:
        shape = (
            self.plan.pump_powers_dbm.size,
            self.plan.pump_freqs_hz.size,
            self.plan.flux_values.size,
            frequency.size,
        )
        dims = ("pump_power", "pump_freq", "ifbl", "frequency")
        return xr.Dataset(
            {
                "s21_real": (dims, np.full(shape, np.nan)),
                "s21_imag": (dims, np.full(shape, np.nan)),
                "completed": (
                    dims[:-1],
                    np.zeros(shape[:-1], dtype=np.int8),
                ),
            },
            coords={
                "pump_power": self.plan.pump_powers_dbm,
                "pump_freq": self.plan.pump_freqs_hz,
                "ifbl": self.plan.flux_values,
                "frequency": frequency,
            },
            attrs={
                "kind": "twpa_pump_on_scan",
                "created_utc": _now(),
                "pump_state": 1,
                "yoko_mode": self.plan.yoko_mode,
                "frequency_unit": "Hz",
                "pump_frequency_unit": "Hz",
                "pump_power_unit": "dBm_at_source",
                "flux_unit": "A" if self.plan.yoko_mode == "current" else "V",
            },
        )

    def _validate_reference(self, data: xr.Dataset, frequency: np.ndarray) -> None:
        if not _matches(data["ifbl"], self.plan.flux_values):
            raise ValueError("Existing reference has a different flux axis")
        if not _matches(data["frequency"], frequency):
            raise ValueError("Existing reference has a different frequency axis")

    def _validate_scan(self, data: xr.Dataset, frequency: np.ndarray) -> None:
        expected = {
            "pump_power": self.plan.pump_powers_dbm,
            "pump_freq": self.plan.pump_freqs_hz,
            "ifbl": self.plan.flux_values,
            "frequency": frequency,
        }
        for name, values in expected.items():
            if not _matches(data[name], values):
                raise ValueError(f"Existing scan has a different {name} axis")

    def acquire_reference(
        self, path: str | os.PathLike[str], *, overwrite: bool = False
    ) -> Path:
        """Acquire pump-off S21 at every flux; rerunning resumes missing rows."""

        output = Path(path)
        program, frequency = self._program()
        data = _load(output) if output.exists() and not overwrite else self._reference_template(frequency)
        self._validate_reference(data, frequency)
        self.pump.off()
        try:
            for flux_index, flux in enumerate(self.plan.flux_values):
                if data["completed"].values[flux_index]:
                    continue
                self._set_flux(flux)
                trace = self._trace(program)
                data["s21_real"].values[flux_index] = trace.real
                data["s21_imag"].values[flux_index] = trace.imag
                data["completed"].values[flux_index] = 1
                data.attrs["updated_utc"] = _now()
                _save(data, output)
                print(f"reference [{flux_index + 1}/{len(self.plan.flux_values)}] flux={flux:+.6g}")
        finally:
            self.pump.off()
            if data["completed"].values.any():
                _save(data, output)
        return output

    def acquire_scan(
        self,
        path: str | os.PathLike[str],
        *,
        reference_path: str | os.PathLike[str],
        overwrite: bool = False,
    ) -> Path:
        """Acquire pump power x pump frequency x flux x signal frequency."""

        output = Path(path)
        program, frequency = self._program()
        reference = _load(Path(reference_path))
        self._validate_reference(reference, frequency)
        if not np.all(reference["completed"].values):
            raise RuntimeError("Pump-off reference is incomplete")
        data = _load(output) if output.exists() and not overwrite else self._scan_template(frequency)
        self._validate_scan(data, frequency)

        if np.all(data["completed"].values):
            return output
        self.pump.power = float(self.plan.pump_powers_dbm[0])
        self.pump.frequency = float(self.plan.pump_freqs_hz[0])
        self.pump.on()
        try:
            for power_index, power in enumerate(self.plan.pump_powers_dbm):
                self.pump.power = float(power)
                for flux_index, flux in enumerate(self.plan.flux_values):
                    self._set_flux(flux)
                    for pump_index, pump_frequency in enumerate(self.plan.pump_freqs_hz):
                        index = (power_index, pump_index, flux_index)
                        if data["completed"].values[index]:
                            continue
                        self.pump.frequency = float(pump_frequency)
                        if self.plan.pump_settle_s:
                            time.sleep(self.plan.pump_settle_s)
                        trace = self._trace(program)
                        data["s21_real"].values[index] = trace.real
                        data["s21_imag"].values[index] = trace.imag
                        data["completed"].values[index] = 1
                        data.attrs["updated_utc"] = _now()
                        done, total = int(data["completed"].sum()), data["completed"].size
                        print(
                            f"scan [{done}/{total}] power={power:+.1f} dBm "
                            f"pump={pump_frequency / 1e9:.6f} GHz flux={flux:+.6g}"
                        )
                    _save(data, output)
        finally:
            self.pump.off()
            if data["completed"].values.any():
                _save(data, output)
        return output

    def shutdown(self, *, flux_safe_value: float | None = None) -> None:
        """Idempotent normal/emergency shutdown."""

        self.pump.off()
        if flux_safe_value is not None:
            self._set_flux(flux_safe_value)


def analyze_twpa_run(
    scan_path: str | os.PathLike[str],
    reference_path: str | os.PathLike[str],
    *,
    target_f_min_hz: float,
    target_f_max_hz: float,
    gain_threshold_db: float = 12.0,
    gain_target_db: float = 15.0,
    ripple_limit_db: float = 5.0,
) -> xr.Dataset:
    """Pure offline analysis using the pump-off row at the same flux."""

    scan, reference = _load(Path(scan_path)), _load(Path(reference_path))
    if not _matches(scan["frequency"], reference["frequency"]):
        raise ValueError("Scan and reference frequency axes differ")
    if not _matches(scan["ifbl"], reference["ifbl"]):
        raise ValueError("Scan and reference flux axes differ")

    off_amplitude = np.abs(_s21(reference))
    gain = 20 * np.log10(
        xr.where(off_amplitude > 0, np.abs(_s21(scan)) / off_amplitude, np.nan)
    )
    gain = gain.transpose("pump_power", "pump_freq", "ifbl", "frequency").rename("gain_db")
    band = gain.where(
        (gain.frequency >= target_f_min_hz) & (gain.frequency <= target_f_max_hz),
        drop=True,
    )
    if band.sizes.get("frequency", 0) < 2:
        raise ValueError("Target band contains fewer than two frequency points")

    median = band.median("frequency", skipna=True).rename("median_gain_db")
    p10 = band.quantile(0.10, "frequency", skipna=True).drop_vars("quantile").rename("p10_gain_db")
    p05 = band.quantile(0.05, "frequency", skipna=True).drop_vars("quantile")
    p95 = band.quantile(0.95, "frequency", skipna=True).drop_vars("quantile")
    ripple = (p95 - p05).rename("ripple_db")
    coverage = (band >= gain_threshold_db).mean("frequency").rename("coverage_fraction")
    raw_score = (
        40 * coverage
        + 25 * (median / gain_target_db).clip(min=0, max=1.5)
        + 20 * (p10 / gain_threshold_db).clip(min=0, max=1.5)
        + 15 * (1 - ripple / ripple_limit_db).clip(min=-1, max=1)
    ).rename("raw_score")
    windows = {
        dim: 3 for dim in ("pump_freq", "ifbl") if raw_score.sizes.get(dim, 0) >= 3
    }
    robust_score = (
        raw_score.rolling(
            windows, center=True, min_periods=int(np.prod(tuple(windows.values())))
        ).mean()
        if windows
        else raw_score
    ).rename("robust_score")
    return xr.Dataset(
        {
            "gain_db": gain,
            "median_gain_db": median,
            "p10_gain_db": p10,
            "ripple_db": ripple,
            "coverage_fraction": coverage,
            "raw_score": raw_score,
            "robust_score": robust_score,
        },
        attrs={
            "normalization": "same_flux_pump_on_over_pump_off",
            "scan_path": str(Path(scan_path).resolve()),
            "reference_path": str(Path(reference_path).resolve()),
            "target_f_min_hz": float(target_f_min_hz),
            "target_f_max_hz": float(target_f_max_hz),
            "gain_threshold_db": float(gain_threshold_db),
            "gain_target_db": float(gain_target_db),
            "ripple_limit_db": float(ripple_limit_db),
        },
    )


def rank_twpa_candidates(
    analysis: xr.Dataset, *, count: int = 5, exclusion_radius_steps: int = 1
) -> list[dict[str, float]]:
    """Return separated candidates ranked by neighborhood-averaged score."""

    score = analysis["robust_score"]
    order = np.argsort(np.nan_to_num(score.values, nan=-np.inf).ravel())[::-1]
    selected: list[tuple[int, ...]] = []
    result: list[dict[str, float]] = []
    for flat_index in order:
        index = tuple(int(i) for i in np.unravel_index(flat_index, score.shape))
        if not np.isfinite(score.values[index]):
            continue
        if any(
            max(abs(a - b) for a, b in zip(index, previous)) <= exclusion_radius_steps
            for previous in selected
        ):
            continue
        candidate = {dim: float(score[dim].values[i]) for dim, i in zip(score.dims, index)}
        selector = {dim: candidate[dim] for dim in score.dims}
        for name in (
            "robust_score",
            "raw_score",
            "median_gain_db",
            "p10_gain_db",
            "ripple_db",
            "coverage_fraction",
        ):
            candidate[name] = float(analysis[name].sel(selector))
        selected.append(index)
        result.append(candidate)
        if len(result) == count:
            break
    return result


def plot_twpa_summary(
    analysis: xr.Dataset, candidate: Mapping[str, float] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the score map and gain curve of one candidate."""

    if candidate is None:
        ranked = rank_twpa_candidates(analysis, count=1)
        if not ranked:
            raise RuntimeError("No finite TWPA candidate")
        candidate = ranked[0]
    power = candidate["pump_power"]
    pump_frequency = candidate["pump_freq"]
    flux = candidate["ifbl"]
    score = analysis.robust_score.sel(pump_power=power)
    curve = analysis.gain_db.sel(
        pump_power=power, pump_freq=pump_frequency, ifbl=flux
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    image = axes[0].imshow(
        score.transpose("ifbl", "pump_freq"),
        aspect="auto",
        origin="lower",
        extent=[
            float(score.pump_freq.min()) / 1e9,
            float(score.pump_freq.max()) / 1e9,
            float(score.ifbl.min()) / 1e-6,
            float(score.ifbl.max()) / 1e-6,
        ],
    )
    figure.colorbar(image, ax=axes[0], label="Robust score")
    axes[0].scatter(pump_frequency / 1e9, flux / 1e-6, marker="x", color="red")
    axes[0].set(xlabel="Pump frequency (GHz)", ylabel="Flux bias (uA)")
    axes[0].set_title(f"Score at {power:+.1f} dBm source power")
    axes[1].plot(curve.frequency / 1e9, curve)
    axes[1].axhline(analysis.attrs["gain_threshold_db"], color="red", linestyle="--")
    axes[1].axvspan(
        analysis.attrs["target_f_min_hz"] / 1e9,
        analysis.attrs["target_f_max_hz"] / 1e9,
        color="gray",
        alpha=0.15,
    )
    axes[1].set(xlabel="Signal frequency (GHz)", ylabel="Gain (dB)")
    axes[1].set_title(
        f"pump={pump_frequency / 1e9:.4f} GHz, flux={flux / 1e-6:.1f} uA"
    )
    axes[1].grid(alpha=0.3)
    return figure, axes


__all__ = [
    "TWPACalibrator",
    "TWPASweepPlan",
    "analyze_twpa_run",
    "latest_twpa_run_directory",
    "new_twpa_run_directory",
    "plot_twpa_summary",
    "rank_twpa_candidates",
]
