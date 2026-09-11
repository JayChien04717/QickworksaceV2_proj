"""Callable-based numerical solver; no model selection or experiment rules."""
import inspect
import warnings
from collections.abc import Mapping
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from .result import FitResult

def _fit_data(x, y, model, min_points=8):
    x, y = np.asarray(x, float), np.asarray(y, float)
    message = None
    if x.ndim != 1 or y.shape != x.shape or len(x) < min_points:
        message = f"At least {min_points} one-dimensional data points are required"
    elif not np.isfinite(x).all() or not np.isfinite(y).all():
        message = "Nonfinite measurement values"
    else:
        order = np.argsort(x)
        x, y = x[order], y[order]
        if np.any(np.diff(x) <= 0):
            message = "Sweep coordinates must be distinct"
        elif np.ptp(y) <= np.finfo(float).eps * max(np.max(np.abs(y)), 1) * 100:
            message = "No measurable contrast"
    return x, y, FitResult(model, False, message=message) if message else None


def fit_curve(
    formula,
    x,
    y,
    *,
    p0=None,
    bounds=None,
    starts=None,
    default_bounds=None,
    model=None,
    units=None,
    min_points=8,
    min_r_squared=0.8,
    maxfev=12000,
):
    """Generic numerical solver; accepts a formula callable, never a model name.

    Supply p0 for a custom formula. Model-specific fit functions provide candidate
    starts and default bounds. No physical model or calibration logic lives here.
    """
    if not callable(formula):
        raise TypeError("fit_curve expects a formula callable; use fit_exponential(), fit_rabi(), etc.")
    model = formula.__name__ if model is None else model
    names = list(inspect.signature(formula).parameters)[1:]
    x, y, failure = _fit_data(x, y, model, max(min_points, len(names) + 1))
    if failure is not None:
        return failure
    if starts is None:
        if p0 is None or isinstance(p0, Mapping) and set(p0) != set(names):
            raise ValueError(f"Supply a full p0 for {model}: {names}")
        starts = [[p0[name] for name in names] if isinstance(p0, Mapping) else list(p0)]
    if default_bounds is None:
        default_bounds = ([-np.inf] * len(names), [np.inf] * len(names))
    starts, limits = _overrides(names, starts, default_bounds, p0, bounds)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        for initial in starts:
            try:
                params, covariance = curve_fit(formula, x, y, p0=initial, bounds=limits, maxfev=maxfev)
                residual = y - formula(x, *params)
                loss = float(residual @ residual)
                if np.isfinite(params).all() and (best is None or loss < best[0]):
                    best = loss, params, covariance, residual
            except (ValueError, RuntimeError, OptimizeWarning, FloatingPointError):
                continue
    if best is None:
        return FitResult(model, False, message="Optimizer could not find a finite solution")
    loss, params, covariance, residual = best
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    r2 = float(1 - loss / np.sum((y - np.mean(y)) ** 2))
    reasons = []
    if r2 < min_r_squared:
        reasons.append(f"R² {r2:.3f} below {min_r_squared:.3f}")
    if not np.isfinite(covariance).all():
        reasons.append("Uncertainty could not be estimated")
    x_fit = np.linspace(x[0], x[-1], max(300, len(x)))
    return FitResult(
        model,
        not reasons,
        dict(zip(names, map(float, params))),
        dict(zip(names, (float(v) if np.isfinite(v) else None for v in errors))),
        dict(units or {}),
        r2,
        "; ".join(reasons) or "Fit passed quality checks",
        x_fit.tolist(),
        formula(x_fit, *params).tolist(),
        residual.tolist(),
    )


def _overrides(names, starts, bounds, p0, parameter_bounds):
    """Resolve named overrides without positional parameter guessing."""
    if p0 is not None and not isinstance(p0, Mapping):
        array = np.asarray(p0, float)
        if array.shape != (len(names),):
            raise ValueError(f"p0 must contain {len(names)} values: {names}")
        starts = [array.tolist()]
    elif p0:
        if set(p0) - set(names):
            raise ValueError(f"Unknown p0 parameters: {set(p0) - set(names)}; available: {names}")
        starts = [[p0.get(name, value) for name, value in zip(names, start)] for start in starts]
    lower, upper = [list(side) for side in bounds]
    if isinstance(parameter_bounds, Mapping):
        if set(parameter_bounds) - set(names):
            raise ValueError(f"Unknown bound parameters: {set(parameter_bounds) - set(names)}")
        for name, (lo, hi) in parameter_bounds.items():
            lower[names.index(name)], upper[names.index(name)] = lo, hi
    elif parameter_bounds is not None:
        lower, upper = parameter_bounds
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != (len(names),) or upper.shape != lower.shape or not np.all(lower < upper):
        raise ValueError("Every lower bound must be smaller than its upper bound")
    starts = [np.asarray(initial, float) for initial in starts]
    if any(not np.isfinite(initial).all() for initial in starts):
        raise ValueError("Initial parameters must be finite")
    if isinstance(p0, Mapping) and p0:
        for name, value in p0.items():
            if not lower[names.index(name)] <= value <= upper[names.index(name)]:
                raise ValueError(f"p0[{name}] is outside its bounds")
    elif p0 is not None and not isinstance(p0, Mapping):
        if np.any(starts[0] < lower) or np.any(starts[0] > upper):
            raise ValueError("p0 is outside its bounds")
    return [np.clip(initial, lower, upper) for initial in starts], (lower, upper)
