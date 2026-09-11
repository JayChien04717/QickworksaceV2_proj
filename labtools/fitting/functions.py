"""Editable equation/fit-function pairs for quantum experiments.

Each fit_<model>() owns its starting values, bounds, derived metrics and quality
checks. fit_curve() only solves a supplied callable and packages diagnostics.
Edit a model's pair below; FIT_OPTIONS and run_cfg['fit'] provide overrides.
Phases are radians, times us and frequencies MHz. Raw data is never changed.
"""

from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln, xlogy
from .result import FitResult
from .solver import fit_curve, _fit_data, _overrides




def _fit_options(model, overrides):
    options = dict(FIT_OPTIONS[model])
    options.update({key: value for key, value in overrides.items() if value is not None})
    return options


def _review(fit, reasons):
    if reasons:
        reasons = ([fit.message] if not fit.success and fit.message else []) + reasons
        fit.success, fit.message = False, "; ".join(reasons)
    return fit


def _uncertain(fit, names):
    return [
        f"{name} is poorly constrained"
        for name in names
        if name in fit.parameters
        and (fit.errors.get(name) is None or fit.errors[name] > abs(fit.parameters[name]) * 0.5)
    ]


def _fft_frequencies(x, y):
    """Candidate starts only; the optimizer always fits original coordinates."""
    span = np.ptp(x)
    uniform_y = np.interp(np.linspace(x[0], x[-1], len(x)), x, y - np.mean(y))
    spectrum = np.abs(np.fft.rfft(uniform_y))
    frequency = np.fft.rfftfreq(len(x), span / (len(x) - 1))
    candidates = np.argsort(spectrum[1:])[-3:] + 1
    low, high = 0.25 / span, 0.49 / np.max(np.diff(x))
    return [np.clip(frequency[i], low * 1.1, high * 0.9) for i in candidates], low, high






# Exponential


def exponential(x, offset, amplitude, tau):
    return offset + amplitude * np.exp(-x / tau)


def fit_exponential(x, y, *, unit="", **options):
    """Exponential fit: edit tau starts, bounds and decay acceptance here."""
    x, y, failure = _fit_data(x, y, "exponential")
    if failure is not None:
        return failure
    span, step = np.ptp(x), float(np.median(np.diff(x)))
    starts = [[y[-1], y[0] - y[-1], span / ratio] for ratio in (1, 2, 5)]
    limits = ([-np.inf, -np.inf, step / 10], [np.inf, np.inf, span * 10])
    fit = fit_curve(
        exponential,
        x,
        y,
        model="exponential",
        starts=starts,
        default_bounds=limits,
        units={"offset": "ADC", "amplitude": "ADC", "tau": unit},
        **_fit_options("exponential", options),
    )
    reasons = _uncertain(fit, ("tau",))
    if fit.parameters.get("tau", 0) > span * 3:
        reasons.append("Sweep is too short to resolve the decay")
    return _review(fit, reasons)


# Lorentzian


def lorentzian(x, offset, amplitude, center, width):
    return offset + amplitude / (1 + ((x - center) / width) ** 2)


def fit_lorentzian(x, y, *, unit="", **options):
    """Lorentzian fit with width uncertainty and resonance-boundary checks."""
    x, y, failure = _fit_data(x, y, "lorentzian")
    if failure is not None:
        return failure
    span, step = np.ptp(x), float(np.median(np.diff(x)))
    baseline = float((y[0] + y[-1]) / 2)
    peak = int(np.argmax(np.abs(y - baseline)))
    starts = [[baseline, y[peak] - baseline, x[peak], span / ratio] for ratio in (5, 10, 20)]
    limits = ([-np.inf, -np.inf, x[0], step / 10], [np.inf, np.inf, x[-1], span * 2])
    fit = fit_curve(
        lorentzian,
        x,
        y,
        model="lorentzian",
        starts=starts,
        default_bounds=limits,
        units={"offset": "ADC", "amplitude": "ADC", "center": unit, "width": unit},
        **_fit_options("lorentzian", options),
    )
    reasons = _uncertain(fit, ("width",))
    if fit.parameters:
        center, width = fit.parameters["center"], fit.parameters["width"]
        if min(center - x[0], x[-1] - center) < width:
            reasons.append("Resonance too close to sweep boundary")
        fit.parameters["fwhm"] = 2 * width
        fit.errors["fwhm"] = 2 * fit.errors["width"] if fit.errors["width"] is not None else None
        fit.units["fwhm"] = unit
    return _review(fit, reasons)


# Asymmetric Lorentzian


def asymmetric_lorentzian(x, offset, amplitude, center, width, asymmetry):
    u = (x - center) / width
    return offset + amplitude * (1 + asymmetry * u) / (1 + u**2)


def fit_asymmetric_lorentzian(x, y, *, unit="", **options):
    """Asymmetric peak fit with its own asymmetry limits."""
    x, y, failure = _fit_data(x, y, "asymmetric_lorentzian")
    if failure is not None:
        return failure
    span, step = np.ptp(x), float(np.median(np.diff(x)))
    baseline = float((y[0] + y[-1]) / 2)
    peak = int(np.argmax(np.abs(y - baseline)))
    starts = [[baseline, y[peak] - baseline, x[peak], span / ratio, 0] for ratio in (5, 10, 20)]
    limits = ([-np.inf, -np.inf, x[0], step / 10, -10], [np.inf, np.inf, x[-1], span * 2, 10])
    fit = fit_curve(
        asymmetric_lorentzian,
        x,
        y,
        model="asymmetric_lorentzian",
        starts=starts,
        default_bounds=limits,
        units={"offset": "ADC", "amplitude": "ADC", "center": unit, "width": unit},
        **_fit_options("asymmetric_lorentzian", options),
    )
    return _review(fit, _uncertain(fit, ("width",)))


# Rabi


def oscillation(x, offset, amplitude, frequency, phase):
    return offset + amplitude * np.cos(2 * np.pi * frequency * x + phase)


def fit_rabi(x, y, *, unit="", **options):
    """Rabi fit: candidate oscillations, pi gains and phase checks are local."""
    x, y, failure = _fit_data(x, y, "rabi")
    if failure is not None:
        return failure
    frequencies, low, high = _fft_frequencies(x, y)
    starts = [
        [float(np.median(y)), float(np.ptp(y)) / 2, f, phase]
        for f in frequencies
        for phase in (0, np.pi / 2, -np.pi / 2, np.pi)
    ]
    limits = ([-np.inf, 0, low, -4 * np.pi], [np.inf, np.inf, high, 4 * np.pi])
    fit = fit_curve(
        oscillation,
        x,
        y,
        model="rabi",
        starts=starts,
        default_bounds=limits,
        units={
            "offset": "ADC",
            "amplitude": "ADC",
            "frequency": "MHz" if unit == "us" else f"1/{unit}" if unit else "1/gain",
            "phase": "rad",
        },
        **_fit_options("rabi", options),
    )
    reasons = _uncertain(fit, ("frequency",))
    if fit.parameters:
        values, errors = fit.parameters, fit.errors
        values["pi_gain"], values["pi2_gain"] = 0.5 / values["frequency"], 0.25 / values["frequency"]
        errors["pi_gain"] = 0.5 * (errors["frequency"] or 0) / values["frequency"] ** 2
        errors["pi2_gain"] = errors["pi_gain"] / 2
        fit.units.update(pi_gain="normalized gain", pi2_gain="normalized gain")
        if values["pi_gain"] > max(x) or values["pi_gain"] <= 0:
            reasons.append("Pi pulse lies outside the measured gain interval")
        phase_error = min(
            abs((values["phase"] + np.pi) % (2 * np.pi) - np.pi), abs(values["phase"] % (2 * np.pi) - np.pi)
        )
        if phase_error > 0.35:
            reasons.append("Rabi phase is inconsistent with a zero-gain preparation")
    return _review(fit, reasons)


# Ramsey


def damped_oscillation(x, offset, amplitude, tau, frequency, phase):
    return offset + amplitude * np.exp(-x / tau) * np.cos(2 * np.pi * frequency * x + phase)


def fit_ramsey(x, y, *, unit="", **options):
    """Ramsey fit: damped-oscillation starts, bounds and decay checks."""
    x, y, failure = _fit_data(x, y, "ramsey")
    if failure is not None:
        return failure
    span, step = np.ptp(x), float(np.median(np.diff(x)))
    frequencies, low, high = _fft_frequencies(x, y)
    starts = [
        [float(np.median(y)), float(np.ptp(y)) / 2, span, f, phase]
        for f in frequencies
        for phase in (0, np.pi / 2, -np.pi / 2, np.pi)
    ]
    limits = ([-np.inf, 0, step / 10, low, -4 * np.pi], [np.inf, np.inf, span * 10, high, 4 * np.pi])
    fit = fit_curve(
        damped_oscillation,
        x,
        y,
        model="ramsey",
        starts=starts,
        default_bounds=limits,
        units={
            "offset": "ADC",
            "amplitude": "ADC",
            "tau": unit,
            "frequency": "MHz" if unit == "us" else f"1/{unit}" if unit else "1/gain",
            "phase": "rad",
        },
        **_fit_options("ramsey", options),
    )
    reasons = _uncertain(fit, ("tau", "frequency"))
    if fit.parameters.get("tau", 0) > span * 3:
        reasons.append("Sweep is too short to resolve the decay")
    return _review(fit, reasons)


# Ramsey Slope


def sloped_damped_oscillation(x, offset, amplitude, tau, frequency, phase, slope):
    return damped_oscillation(x, offset, amplitude, tau, frequency, phase) + slope * x


def fit_ramsey_slope(x, y, *, unit="", **options):
    """Fit ramsey_slope; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        sloped_damped_oscillation,
        x,
        y,
        model="ramsey_slope",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("ramsey_slope", options),
    )
    reasons = _uncertain(fit, ("tau", "frequency"))
    if "tau" in fit.parameters and fit.parameters["tau"] > np.ptp(x) * 3:
        reasons.append("Sweep is too short to resolve the decay")
    return _review(fit, reasons)


# Ramsey Two Frequency


def two_frequency_damped_oscillation(
    x, offset, amplitude, tau, frequency, phase, amplitude2, frequency2, phase2
):
    return offset + np.exp(-x / tau) * (
        amplitude * np.cos(2 * np.pi * frequency * x + phase)
        + amplitude2 * np.cos(2 * np.pi * frequency2 * x + phase2)
    )


def fit_ramsey_two_frequency(x, y, *, unit="", **options):
    """Fit ramsey_two_frequency; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        two_frequency_damped_oscillation,
        x,
        y,
        model="ramsey_two_frequency",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("ramsey_two_frequency", options),
    )
    reasons = _uncertain(fit, ("tau", "frequency"))
    if "tau" in fit.parameters and fit.parameters["tau"] > np.ptp(x) * 3:
        reasons.append("Sweep is too short to resolve the decay")
    return _review(fit, reasons)


# Gaussian


def gaussian(x, offset, amplitude, center, sigma):
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def fit_gaussian(x, y, *, unit="", **options):
    """Gaussian peak fit; sigma starts and bounds are maintained here."""
    x, y, failure = _fit_data(x, y, "gaussian")
    if failure is not None:
        return failure
    span, step = np.ptp(x), float(np.median(np.diff(x)))
    baseline = float((y[0] + y[-1]) / 2)
    peak = int(np.argmax(np.abs(y - baseline)))
    starts = [[baseline, y[peak] - baseline, x[peak], span / ratio] for ratio in (5, 10, 20)]
    limits = ([-np.inf, -np.inf, x[0], step / 10], [np.inf, np.inf, x[-1], span * 2])
    return fit_curve(
        gaussian,
        x,
        y,
        model="gaussian",
        starts=starts,
        default_bounds=limits,
        units={"offset": "ADC", "amplitude": "ADC", "center": unit, "sigma": unit},
        **_fit_options("gaussian", options),
    )


# Double Gaussian


def double_gaussian(x, amplitude_g, center_g, sigma_g, amplitude_e, center_e, sigma_e):
    return gaussian(x, 0, amplitude_g, center_g, sigma_g) + gaussian(x, 0, amplitude_e, center_e, sigma_e)


def fit_double_gaussian(x, y, *, unit="", **options):
    """Fit double_gaussian; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        double_gaussian,
        x,
        y,
        model="double_gaussian",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("double_gaussian", options),
    )
    return fit


# Rb


def rb_decay(depth, offset, amplitude, probability):
    return offset + amplitude * probability**depth


def fit_rb(x, y, *, unit="", **options):
    """RB sequence-mean fit with local probability/EPC uncertainty checks."""
    x, y, failure = _fit_data(x, y, "rb", min_points=5)
    if failure is not None:
        return failure
    starts = [[y[-1], y[0] - y[-1], probability] for probability in (0.9, 0.99)]
    limits = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 1])
    fit = fit_curve(
        rb_decay,
        x,
        y,
        model="rb",
        starts=starts,
        default_bounds=limits,
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("rb", options),
    )
    reasons = []
    if fit.parameters:
        probability, error = fit.parameters.pop("probability"), fit.errors.pop("probability")
        fit.parameters.update(decay_probability=probability, error_per_clifford=(1 - probability) / 2)
        fit.errors.update(decay_probability=error, error_per_clifford=None if error is None else error / 2)
        if error is None or error >= max(1 - probability, 1e-8):
            reasons.append("RB decay probability is poorly constrained")
    return _review(fit, reasons)


# Rotation X


def rotation_x(n, offset, error_rad):
    return offset + 0.5 * np.cos(np.pi / 2 + 2 * np.asarray(n) * error_rad)


def fit_rotation_x(x, y, *, unit="", **options):
    """Fit rotation_x; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        rotation_x,
        x,
        y,
        model="rotation_x",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("rotation_x", options),
    )
    return fit


# Rotation X Half


def rotation_x_half(n, offset, error_rad):
    return offset + 0.5 * np.cos(np.pi * np.asarray(n)) * np.cos(np.pi / 2 + 2 * np.asarray(n) * error_rad)


def fit_rotation_x_half(x, y, *, unit="", **options):
    """Fit rotation_x_half; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        rotation_x_half,
        x,
        y,
        model="rotation_x_half",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("rotation_x_half", options),
    )
    return fit


# Rotation X Half Decay


def rotation_x_half_decay(n, offset, error_rad, tau):
    return offset + (rotation_x_half(n, 0, error_rad)) * np.exp(-np.asarray(n) / tau)


def fit_rotation_x_half_decay(x, y, *, unit="", **options):
    """Fit rotation_x_half_decay; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        rotation_x_half_decay,
        x,
        y,
        model="rotation_x_half_decay",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("rotation_x_half_decay", options),
    )
    reasons = _uncertain(fit, ("tau",))
    if "tau" in fit.parameters and fit.parameters["tau"] > np.ptp(x) * 3:
        reasons.append("Sweep is too short to resolve the decay")
    return _review(fit, reasons)


# Poisson


def poisson(n, mean):
    n = np.asarray(n, float)
    return np.exp(xlogy(n, mean) - mean - gammaln(n + 1))


def fit_poisson(x, y, *, unit="", **options):
    """Fit poisson; supply a full named p0 and optional bounds."""
    fit = fit_curve(
        poisson,
        x,
        y,
        model="poisson",
        units={"offset": "ADC", "amplitude": "ADC"},
        **_fit_options("poisson", options),
    )
    return fit


# Complex resonator


def notch_response(
    frequency, center, linewidth, depth, asymmetry=0.0, scale=1.0, phase=0.0, delay_us=0.0, reference_mhz=None
):
    f = np.asarray(frequency, float)
    reference = float(np.mean(f)) if reference_mhz is None else reference_mhz
    background = scale * np.exp(1j * phase - 2j * np.pi * (f - reference) * delay_us)
    return background * (1 - depth * np.exp(1j * asymmetry) / (1 + 2j * (f - center) / linewidth))


def fit_resonator(frequency_mhz, iq, *, unit="MHz", min_r_squared=None, p0=None, bounds=None, maxfev=None):
    """Fit complex S21; return Ql, |Qc| and diameter-corrected Qi with uncertainties."""
    settings = FIT_OPTIONS["complex_notch"]
    min_r_squared = settings["min_r_squared"] if min_r_squared is None else min_r_squared
    maxfev = settings["maxfev"] if maxfev is None else maxfev
    p0 = settings["p0"] if p0 is None else p0
    user_bounds = settings["bounds"] if bounds is None else bounds
    f, z = np.asarray(frequency_mhz, float), np.asarray(iq, complex)
    fail = lambda message: FitResult("complex_notch", False, message=message)
    if f.ndim != 1 or z.shape != f.shape or len(f) < 16:
        return fail("At least 16 complex frequency points required")
    if not np.isfinite(f).all() or not np.isfinite(z).all():
        return fail("Nonfinite resonator data")
    order = np.argsort(f)
    f, z = f[order], z[order]
    if np.any(np.diff(f) <= 0) or np.ptp(abs(z)) < 1e-10:
        return fail("Distinct frequencies and measurable notch contrast required")
    reference, span = float(np.mean(f)), float(np.ptp(f))
    x = (f - reference) / span
    scale = float(np.median(np.r_[abs(z[: len(z) // 8]), abs(z[-len(z) // 8 :])]))
    y = z / scale
    # Initial slope uses both ends where the resonance contribution is small.
    edges = np.r_[np.arange(len(f) // 8), np.arange(len(f) - len(f) // 8, len(f))]
    slope, phase = np.polyfit(x[edges], np.unwrap(np.angle(y))[edges], 1)
    delay = -slope / (2 * np.pi)
    center = float(x[np.argmin(abs(y))])
    lower = [-0.5, 1 / (len(f) * 10), 0.001, -1.5, 0.01, phase - 2 * np.pi, delay - 10]
    upper = [0.5, 2.0, 3.0, 1.5, 10.0, phase + 2 * np.pi, delay + 10]

    def model(p, axis=x):
        c, width, depth, asym, amp, angle, d = p
        return notch_response(axis, c, width, depth, asym, amp, angle, d, reference_mhz=0)

    def residual(p):
        r = model(p) - y
        return np.r_[r.real, r.imag]

    names = ["center", "linewidth", "depth", "asymmetry_rad", "scale", "phase_rad", "delay_us"]

    # Optimize normalized coordinates, but accept editable values in physical units.
    def normalized(name, value):
        if name == "center":
            return (value - reference) / span
        if name == "linewidth":
            return value / span
        if name == "scale":
            return value / scale
        if name == "delay_us":
            return value * span
        return value

    if not isinstance(p0, Mapping) or not isinstance(user_bounds, Mapping):
        raise ValueError("Complex notch p0 and bounds use named physical parameters")
    guesses = [
        [center, width, max(0.05, 1 - min(abs(y))), asym, 1.0, phase, delay]
        for width in (0.03, 0.1, 0.3)
        for asym in (-0.2, 0, 0.2)
    ]
    guesses, fit_bounds = _overrides(
        names,
        guesses,
        (lower, upper),
        {key: normalized(key, value) for key, value in p0.items()},
        {key: tuple(normalized(key, value) for value in pair) for key, pair in user_bounds.items()},
    )
    attempts = []
    for initial in guesses:
        try:
            attempts.append(least_squares(residual, initial, bounds=fit_bounds, max_nfev=maxfev))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if not attempts:
        return fail("Complex notch optimizer could not find a finite solution")
    best = min(attempts, key=lambda fit: np.sum(fit.fun**2))
    c, width, depth, asym, amp, angle, d = best.x
    center_mhz, linewidth = reference + c * span, width * span
    ql, inverse_fraction = center_mhz / linewidth, 1 - depth * np.cos(asym)
    r2 = float(1 - np.sum(best.fun**2) / np.sum(abs(y - y.mean()) ** 2))
    reasons = []
    if not best.success or r2 < min_r_squared:
        reasons.append("Complex fit did not pass convergence/contrast quality")
    if min(center_mhz - f[0], f[-1] - center_mhz) < linewidth:
        reasons.append("Notch too close to boundary; include both baseline regions")
    if inverse_fraction <= 0:
        reasons.append("Nonphysical internal loss; check notch model and calibration")
    params = dict(
        center=center_mhz,
        linewidth=linewidth,
        Q_loaded=ql,
        Q_coupling_abs=ql / depth,
        depth=depth,
        asymmetry_rad=asym,
        scale=amp * scale,
        phase_rad=angle,
        delay_us=d / span,
    )
    if inverse_fraction > 0:
        params["Q_internal"] = ql / inverse_fraction
    errors = {}
    if np.linalg.matrix_rank(best.jac) < len(best.x):
        reasons.append("Resonator parameters are not identifiable")
    else:
        covariance = np.linalg.pinv(best.jac.T @ best.jac) * np.sum(best.fun**2) / (2 * len(f) - len(best.x))

        def metrics(p):
            c, w, dep, a, amp, ph, delay = p
            q = (reference + c * span) / (w * span)
            values = [reference + c * span, w * span, q, q / dep, dep, a, amp * scale, ph, delay / span]
            if "Q_internal" in params:
                values.append(q / (1 - dep * np.cos(a)))
            return np.array(values)

        jac = np.empty((len(params), len(best.x)))
        for i in range(len(best.x)):
            step = 1e-6 * max(abs(best.x[i]), 1)
            plus, minus = best.x.copy(), best.x.copy()
            plus[i] += step
            minus[i] -= step
            jac[:, i] = (metrics(plus) - metrics(minus)) / (2 * step)
        std = np.sqrt(np.maximum(np.diag(jac @ covariance @ jac.T), 0))
        errors = dict(zip(params, map(float, std)))
        for key in ("linewidth", "Q_internal"):
            if key in errors and (not np.isfinite(errors[key]) or errors[key] > abs(params[key]) * 0.5):
                reasons.append(f"{key} is poorly constrained")
        errors = {k: v if np.isfinite(v) else None for k, v in errors.items()}
    xf = np.linspace(f[0], f[-1], 400)
    return FitResult(
        "complex_notch",
        not reasons,
        {k: float(v) for k, v in params.items()},
        errors,
        {"center": "MHz", "linewidth": "MHz", "delay_us": "us"},
        r2,
        "; ".join(reasons) or "Complex notch fit passed; Qi assumes the hanger transmission model",
        xf.tolist(),
        (abs(model(best.x, (xf - reference) / span)) * scale).tolist(),
        (abs(z) - abs(model(best.x)) * scale).tolist(),
    )


# Only configuration selection lives here; each function above owns its model.
FIT_FUNCTIONS = {
    "exponential": fit_exponential,
    "lorentzian": fit_lorentzian,
    "asymmetric_lorentzian": fit_asymmetric_lorentzian,
    "rabi": fit_rabi,
    "ramsey": fit_ramsey,
    "ramsey_slope": fit_ramsey_slope,
    "ramsey_two_frequency": fit_ramsey_two_frequency,
    "gaussian": fit_gaussian,
    "double_gaussian": fit_double_gaussian,
    "rb": fit_rb,
    "rotation_x": fit_rotation_x,
    "rotation_x_half": fit_rotation_x_half,
    "rotation_x_half_decay": fit_rotation_x_half_decay,
    "poisson": fit_poisson,
    "complex_notch": fit_resonator,
}

# Optional global overrides. Data-derived starting values/bounds live beside
# each equation in its fit function, not in the generic solver.
FIT_OPTIONS = {
    name: {"p0": {}, "bounds": {}, "maxfev": 12000, "min_r_squared": 0.8} for name in FIT_FUNCTIONS
}
FIT_OPTIONS["rb"].update(maxfev=20000, min_points=5)
FIT_OPTIONS["complex_notch"].update(maxfev=3000, min_r_squared=0.95)
