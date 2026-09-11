"""Pure array and quality helpers shared by mux experiments."""

from __future__ import annotations

import numpy as np

from ..core.experiment_data import QualityFlag


def project_iq(iq_data, mode: str = "abs") -> np.ndarray:
    """Project complex IQ data onto a named real-valued display channel."""
    values = np.asarray(iq_data)
    channel = (mode or "abs").lower()
    if channel in {"real", "i", "avgi"}:
        return np.real(values)
    if channel in {"imag", "q", "avgq"}:
        return np.imag(values)
    if channel == "phase":
        return np.unwrap(np.angle(values), axis=-1)
    return np.abs(values)


def fit_snr(measured, fitted) -> float:
    """Return fitted span divided by residual noise."""
    measured = np.asarray(measured, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    residual_noise = float(np.nanstd(measured - fitted))
    fitted_span = float(np.nanmax(fitted) - np.nanmin(fitted))
    return fitted_span / max(residual_noise, 1e-12)


def fit_quality(
    has_data: bool,
    successful_fits: int,
    expected_fits: int,
    label: str,
) -> tuple[QualityFlag, str]:
    """Assess acquisition and fitting separately for a mux result."""
    if not has_data:
        return QualityFlag.BAD, "No data acquired."
    if successful_fits == 0:
        return QualityFlag.BAD, f"{label} acquired, but all fits failed."
    if successful_fits < expected_fits:
        return (
            QualityFlag.WARNING,
            f"{label} acquired; {successful_fits}/{expected_fits} fits succeeded.",
        )
    return QualityFlag.GOOD, f"{label} acquired and all fits succeeded."
