"""Reusable fits with bounded parameters, uncertainty and explicit failure."""

from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from QickworkspaceV2.data.models import FitResult


def lorentzian(x, offset, amplitude, center, width):
    return offset + amplitude / (1 + ((x - center) / width) ** 2)


def exponential(x, offset, amplitude, tau):
    return offset + amplitude * np.exp(-x / tau)


def oscillation(x, offset, amplitude, frequency, phase):
    return offset + amplitude * np.cos(2 * np.pi * frequency * x + phase)


def damped_oscillation(x, offset, amplitude, tau, frequency, phase):
    return offset + amplitude * np.exp(-x / tau) * np.cos(2 * np.pi * frequency * x + phase)


MODELS = {
    "lorentzian": lorentzian,
    "exponential": exponential,
    "rabi": oscillation,
    "ramsey": damped_oscillation,
}


def fit_curve(model, x, y, *, unit="", min_r_squared=0.8):
    """Fit one trace. Flat, underdetermined or nonfinite input fails visibly."""
    if model not in MODELS:
        raise KeyError(f"Unknown fit model {model}")
    x, y = np.asarray(x, float), np.asarray(y, float)
    fail = lambda msg: FitResult(model, False, message=msg)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 8:
        return fail("At least eight one-dimensional data points are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return fail("Nonfinite measurement values")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        return fail("Sweep coordinates must be distinct")
    span, scale = np.ptp(x), np.ptp(y)
    if scale <= np.finfo(float).eps * max(np.max(np.abs(y)), 1) * 100:
        return fail("No measurable contrast")
    offset, amp = float(np.median(y)), float(scale)
    step = float(np.median(np.diff(x)))
    starts = []
    units = {"offset": "ADC", "amplitude": "ADC"}
    if model == "lorentzian":
        names = ["offset", "amplitude", "center", "width"]
        baseline = float((y[0] + y[-1]) / 2)
        peak = int(np.argmax(np.abs(y - baseline)))
        starts = [[baseline, y[peak] - baseline, x[peak], span / ratio] for ratio in (5, 10, 20)]
        bounds = ([-np.inf, -np.inf, x[0], step / 10], [np.inf, np.inf, x[-1], span * 2])
        units.update(center=unit, width=unit)
    elif model == "exponential":
        names = ["offset", "amplitude", "tau"]
        starts = [[y[-1], y[0] - y[-1], span / ratio] for ratio in (1, 2, 5)]
        bounds = ([-np.inf, -np.inf, step / 10], [np.inf, np.inf, span * 10])
        units["tau"] = unit
    else:
        centered = y - np.mean(y)
        # Resample only for initial frequency estimates; fit original coordinates.
        uniform_y = np.interp(np.linspace(x[0], x[-1], len(x)), x, centered)
        spec = np.abs(np.fft.rfft(uniform_y))
        freq_axis = np.fft.rfftfreq(len(x), span / (len(x) - 1))
        candidates = np.argsort(spec[1:])[-3:] + 1
        min_f, max_f = 0.25 / span, 0.49 / np.max(np.diff(x))
        for index in candidates:
            f = np.clip(freq_axis[index], min_f * 1.1, max_f * 0.9)
            for phase in (0, np.pi / 2, -np.pi / 2, np.pi):
                base = [offset, amp / 2]
                starts.append(base + ([span] if model == "ramsey" else []) + [f, phase])
        if model == "rabi":
            names = ["offset", "amplitude", "frequency", "phase"]
            bounds = ([-np.inf, 0, min_f, -4 * np.pi], [np.inf, np.inf, max_f, 4 * np.pi])
        else:
            names = ["offset", "amplitude", "tau", "frequency", "phase"]
            bounds = (
                [-np.inf, 0, step / 10, min_f, -4 * np.pi],
                [np.inf, np.inf, span * 10, max_f, 4 * np.pi],
            )
            units["tau"] = unit
        units.update(frequency="MHz" if unit == "us" else f"1/{unit}" if unit else "1/gain", phase="rad")
    fn, best = MODELS[model], None
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        for initial in starts:
            try:
                params, covariance = curve_fit(fn, x, y, p0=initial, bounds=bounds, maxfev=12000)
                residual = y - fn(x, *params)
                loss = float(residual @ residual)
                if np.isfinite(params).all() and (best is None or loss < best[0]):
                    best = loss, params, covariance, residual
            except (ValueError, RuntimeError, OptimizeWarning, FloatingPointError):
                continue
    if best is None:
        return fail("Optimizer could not find a finite solution")
    loss, params, covariance, residual = best
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    r2 = float(1 - loss / np.sum((y - np.mean(y)) ** 2))
    values = dict(zip(names, map(float, params)))
    uncertainty = dict(zip(names, (float(v) if np.isfinite(v) else None for v in errors)))
    reasons = []
    if r2 < min_r_squared:
        reasons.append(f"R² {r2:.3f} below {min_r_squared:.3f}")
    if not np.isfinite(covariance).all():
        reasons.append("Uncertainty could not be estimated")
    for key in ("tau", "width", "frequency"):
        if key in values and (uncertainty[key] is None or uncertainty[key] > abs(values[key]) * 0.5):
            reasons.append(f"{key} is poorly constrained")
    if "tau" in values and values["tau"] > span * 3:
        reasons.append("Sweep is too short to resolve the decay")
    if model == "lorentzian":
        if min(values["center"] - x[0], x[-1] - values["center"]) < values["width"]:
            reasons.append("Resonance too close to sweep boundary")
        values["fwhm"] = 2 * values["width"]
        uncertainty["fwhm"] = 2 * uncertainty["width"] if uncertainty["width"] is not None else None
        units["fwhm"] = unit
    if model == "rabi":
        values["pi_gain"] = 0.5 / values["frequency"]
        values["pi2_gain"] = 0.25 / values["frequency"]
        uncertainty["pi_gain"] = 0.5 * (uncertainty["frequency"] or 0) / values["frequency"] ** 2
        uncertainty["pi2_gain"] = uncertainty["pi_gain"] / 2
        units.update(pi_gain="normalized gain", pi2_gain="normalized gain")
        if values["pi_gain"] > max(x) or values["pi_gain"] <= 0:
            reasons.append("Pi pulse lies outside the measured gain interval")
        phase_error = min(
            abs((values["phase"] + np.pi) % (2 * np.pi) - np.pi), abs(values["phase"] % (2 * np.pi) - np.pi)
        )
        if phase_error > 0.35:
            reasons.append("Rabi phase is inconsistent with a zero-gain preparation")
    x_fit = np.linspace(x[0], x[-1], max(300, len(x)))
    return FitResult(
        model,
        not reasons,
        values,
        uncertainty,
        units,
        r2,
        "; ".join(reasons) or "Fit passed quality checks",
        x_fit.tolist(),
        fn(x_fit, *params).tolist(),
        residual.tolist(),
    )


def analyze_traces(result, model, *, signal="abs", event=0, min_r_squared=0.8):
    fits = {}
    for q, trace in result.traces.items():
        dims = [d for d in trace.dims if d != "readout"]
        if len(dims) != 1 or "readout" not in trace.dims:
            fits[q] = FitResult(
                model, False, message="Select one sweep axis and one readout event before fitting"
            )
            continue
        dim = dims[0]
        values = trace.signal(signal, rotation_deg=trace.metadata.get("rotation_deg", 0))
        y = np.take(values, event, axis=trace.dims.index("readout"))
        fits[q] = fit_curve(
            model, trace.coords[dim], y, unit=trace.units.get(dim, ""), min_r_squared=min_r_squared
        )
    return fits
